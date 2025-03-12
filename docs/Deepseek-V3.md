# DeepSeek-V3

DeepSeek-R1能有这么强力的表现和DeepSeek-V3这个基模型的强大是分不开的。

[![dsv3](https://www.linsight.cn/a9c496e3/perf.png)](https://www.linsight.cn/a9c496e3/perf.png)

## MLA

<img src="https://www.linsight.cn/a9c496e3/ds3_archi.png" alt="img" style="zoom:50%;" />

### 从MHA出发

标准的MHA， 假设$n_h$ 是注意力头的数量， $d_n$ 是每个注意力头的大小, $\mathbf{h}_{t}\in\mathbb{R}^{d}$ 是第t个输入token.

MHA首先通过三个投影矩阵$W^{Q},W^{K},W^{V}\in\mathbb{R}^{d_{h}n_{h}\times
d}$ 获得 $\mathbf{q}_t,\mathbf{k}_t,\mathbf{v}_t\in\mathbb{R}^{d_hn_h}$ :
$$
\mathbf{q}_t=W^Q\mathbf{h}_t\\
\mathbf{k}_t=W^K\mathbf{h}_t\\
\mathbf{v}_t=W^V\mathbf{h}_t\\
$$
 

之后 $q_t,k_t, v_t$ 就会被切成 $n_h$ 份, 分别进行注意力计算:
$$
[\mathbf{q}_{t,1};\mathbf{q}_{t,2};...;\mathbf{q}_{t,n_{h}}]=\mathbf{q}_{t}\\
[\mathbf{k}_{t,1};\mathbf{k}_{t,2};...;\mathbf{k}_{t,n_{h}}]=\mathbf{k}_{t}\\
[\mathbf{v}_{t,1};\mathbf{v}_{t,2};...;\mathbf{v}_{t,n_{h}}]=\mathbf{v}_{t}\\
\mathbf{o}_{t,i}=\sum_{j=1}^t\mathrm{Softmax}_j(\frac{\mathbf{q}_{t,i}^T\mathbf{k}_{j,i}}{\sqrt{d_h}})\mathbf{v}_{j,i}\\
\mathbf{u}_t=W^O[\mathbf{o}_{t,1};\mathbf{o}_{t,2};...;\mathbf{o}_{t,n_h}]
$$
其中 $\mathbf{q}_{t,i},\mathbf{k}_{t,i},\mathbf{v}_{t,i}\in\mathbb{R}^{d_{h}}$ , $W^O\in\mathbb{R}^{d\times d_hn_h}$ .

在推理的时候, 为了加速, 会对前面已经计算过的K, V值进行缓存, 那么每个token在模型层就要保存$2n_hd_h$ 个数值.

那么要减少缓存量, 一个方法就是减少使用的K/V. GQA/MQA就是通过共享参数减少K, V头的数量并重复使用, 从而减少了需要缓存的KV的量.

### MLA

M LA通过对K和V做low-rank joint compression来压缩KV cache, 理论上可以更有效压缩KV缓存值.

![img](https://www.linsight.cn/a9c496e3/ds3_MLA.png)

在MHA中, K和V是对$h_t$ 分别用投影矩阵进行变化得到的, 而MLA把KV的变化改成使用一个共用的down-projection matrix和两个up-projection matrix进行操作.
$$
\mathbf{c}_t^{KV}=W^{DKV}\mathbf{h}_t\\
\mathbf{k}_t^C=W^{UK}\mathbf{c}_t^{KV}\\
\mathbf{v}_t^C=W^{UV}\mathbf{c}_t^{KV}
$$
$\mathfrak{c}_t^{KV}\in\mathbb{R}^{d_c}$ 就是K和V的compressed latent vector, 这也是推理时需要缓存的部份.

这里相当于把MHA中的$W^{K},W^{V}$ 拆成两个矩阵:
$$
\mathbf{k}_t=W^K\mathbf{h}_t\rightarrow
W^{UK}W^{DKV}\mathbf{h}_t\\
\mathbf{v}_t=W^V\mathbf{h}_t\rightarrow
W^{UV}W^{DKV}\mathbf{h}_t
$$
$d_c$ 是KV的压缩维度, 让$d_c\ll d_hn_h$ 就可以大大减少推理时需要缓存的数据量. 

再看回attention计算，在得到q、k、v之后，会计算权重矩阵并获得最终注意力输出结果：
$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}(\frac{Q^TK}{\sqrt{d}})V
$$
而$Q^TK=H^T(W^Q)^TW^{UK}C$ , 因此 $W^{UK}$ 可以被吸收进入 $W^{Q}$ 中, 而不用在计算时显示计算出K, 只需要调整 $W^{Q}$ 的shape后直接输入C即可. 

此外, MLA还对Q也做了low-rank compression, 跟K, V的操作类似:
$$
\mathbf{c}_t^Q=W^{DQ}\mathbf{h}_t,\\
\mathbf{q}_t^C=W^{UQ}\mathbf{c}_t^Q,
$$

### 兼容RoPE

到这里似乎MLA已经完成了，即减少了缓存的量，也不用引入其他overhead（两个up-projection matrices都不用算了）。

但是实际上还有一个问题没有解决：位置编码使用的是RoPE，而RoPE是通过在Q、K上乘一个旋转矩阵来编码位置的。

而在上面MLA的设计中，已经没有显式计算K了，而RoPE也不能加在latent vector上。一个方法是重新把K和V显式计算出来，但是这样计算量就会增加，MLA的推理加速效果就会打折扣。

针对这个问题, 解决方案是使用decoupled RoPE: 使用额外的multi-head queries $\mathbf{q}_{t,i}^R\in\mathbb{R}^{d_h^R}$ 和一个 share key $\mathbf{k}_t^R\in\mathbb{R}^{d_h^R}$ 来携带RoPE的位置信息, $d_h^R$ 是decoupled queries的维度.

新增的q和k维度使用常规的RoPE计算，用于携带位置信息；而原来的维度依然使用低秩分解的方式计算，最后再计算attention的时候两个部分拼接起来。

最终完整的MLA计算如下:

<img src="https://www.linsight.cn/a9c496e3/MLA_formula.png" alt="img" style="zoom:67%;" />

篮框中的部份就是推理时需要缓存的内容.

MLA所需的缓存量约等于组数为2.5的GQA:

![img](https://www.linsight.cn/a9c496e3/MLA_cache.png)

## MoE

DeepSeek-V3的MoE结构设计和DeepSeekMoE/DeepSeek-V2基本一致。和V2相比，有一些设置是一样的：

- 初始化 standard deviation = 0.006
- 128个attention head，head size = 128
- KV的compression dimension dc = 512
- Q的compression dimension dc' = 1536
- decoupled queries and key per head dimension = 64

此外，也有一些具体设置和V2相比有变化：
\- layers = 61（比V2多1层）
\- hidden dimension = 7168（比V2的5120更大）
\- 前3层不使用MoE
\- 1个共享专家 + 8/256个路由专家，专家大小为2048（更多专家，专家维度更大）
\- 每个token最多只会被分发到4个节点
\- 总参数671B，激活参数37B

\- gating在计算affinity score的时候先用sigmoid函数，再在选定的分数上进行归一化，而V2是直接使用softmax

V2的总参数为236B，激活参数为21B；而V3的总参数为671B，激活参数为37B。可以看到相比V2，V3多的参数主要在模型宽度和专家数量，而且MoE的激活更为稀疏。

### 负载平衡

1. Auxiliary-Loss-Free Load Balancing

V3的MoE计算:
$$
\mathbf{h}_t^{\prime}=\mathbf{u}_t+\sum_{i=1}^{N_s}\mathrm{FFN}_i^{(s)}\left(\mathbf{u}_t\right)+\sum_{i=1}^{N_r}g_{i,t}\mathrm{FFN}_i^{(r)}\left(\mathbf{u}_t\right)
$$
第一项来自残差连接, 第二项是共享专家的输出, 第三项是路由专家的输出; Ns是share expert的数量, Nr是routed expert的数量, V3中Ns=1, Nr=128.
$$
g_{i,t}=\frac{g_{i,t}^\prime}{\sum_{j=1}^{N_r}g_{j,t}^\prime}
$$
g'只保留top Nr个（DeepSeek-V3中Nr=8），其他都置零了。
$$
g_{i,t}^{\prime}=\begin{cases}s_{i,t},&s_{i,t}\in\mathrm{Topk}(\{s_{j,t}|1\leqslant
j\leqslant
N_r\},K_r)\\0,&\text{otherwise}&&\end{cases}
$$
Kr是activated routed expert的数量. $s_{i,t}$ 表示第i个路由专家在第t个token的选择信号, 该信号通常用于指示该专家是否被选中参与计算.

之前的版本使用auxiliary loss来对top affinity score的分配不平衡进行惩罚，以此来缓解专家分配不平衡的问题。由于auxiliary loss的设计并不关注模型的效果，因此过大的权重会对模型的训练效果产生损害。

为了避免模型效果的损失，DeepSeek-V3不使用auxiliary loss来平衡负载，而是在**affinity score上加了一个bias term**，这个bias term和expert是一一对应的：
$$
g_{i,t}^{\prime}=\begin{cases}s_{i,t},&s_{i,t}+b_i\in\mathrm{Topk}(\{s_{j,t}+b_j|1\leqslant
j\leqslant
N_r\},K_r)\\0,&\text{otherwise}&\end{cases}
$$
这个bias term只用于routing, 不用于和FFN的结构相乘输出专家的feature vector. 在每个训练step后, 如果一个expert的负载过大, 就会把对应的bias term减少𝛾, 反之则把bias term的数值增大𝛾. 𝛾是个超参, 控制负载平衡系统的变化速度.

2. Complementary Sequence-Wise Auxiliary Loss

虽然加了bias term控制负载均衡, 但为了防止极端不平衡状况的出现, 还是额外加了一个Auxiliary Loss.
$$
\mathcal{L}_\mathrm{Bal}=\alpha\sum_{i=1}^{N_r}f_iP_i\\
P_i=\frac{1}{T}\sum_{t=1}^Ts_{i,t}^{\prime}\\
s_{i,t}^\prime=\frac{s_{i,t}}{\sum_{j=1}^{N_r}s_{j,t}}\\
f_i=\frac{N_r}{K_rT}\sum_{t=1}^T\mathbb{1}\left(s_{i,t}\in\mathrm{Topk}(\{s_{j,t}|1\leqslant
j\leqslant N_r\},K_r)\right)
$$
s`是归一化的affinity score.

求和部分其实就是某个token是否选择了expert i。训练中𝛼 = 0.0001。

fi是不可导的，Pi是可导的。

在完美负载平衡的情况下，affinity score均匀分配，每个expert的得分相同，那么有
$$
P_i=\frac{1}{T}\times T\times
\frac{1}{N_r}=\frac{1}{N_r}\\
f_i=\frac{N_r}{K_rT}\sum_{t=1}^T\frac{K_r}{N_r}=1
$$
那么
$$
\mathcal{L}_\mathrm{Bal}=\alpha\sum_{i=1}^{N_r}\frac{1}{N_r}=\alpha
$$


complementary sequence-wise balance loss其实就是DeepSeekMoE中的expert-level balance loss。

而在极端不平衡的情况下，比如所有token都选择了前Kr个expert激活，那么对于激活的expert i，有
$$
P_i=\frac{1}{T}\times T\times
1=1\\
f_i=\frac{N_r}{K_rT}\sum_{t=1}^T1=\frac{N_r}{K_r}
$$
那么就有:
$$
\mathcal{L}_\mathrm{Bal}=\alpha\sum_{i=1}^{K_r}\frac{N_r}{K_r}=\alpha
N_r
$$

3. Node-Limited Routing

在前面的基础上，最后还加了一个机制，限制每个token最多只能分发到M个节点上，而节点的选择是基于每个节点上的affinity score的总和的。

举个例子，在Kr=8，M=4的情况下：

- 如果8个得分最高的专家都分布在不同的node，那么只有top4个专家会被激活，其余的专家虽然得分排在top Nr，但是由于激活节点的限制，不会被使用；
- top8个专家分配在5个节点上：
    - 节点1：0.1,0.1
    - 节点2：0.1,0.1
    - 节点3：0.1,0.1
    - 节点4：0.25
    - 节点5：0.15
        在这样的情况下，虽然节点5上的专家得分是第二高的，但是由于它所在的节点的得分总和不高，因此不会被激活

### No Token-Dropping

由于前面的几个负载平衡策略基本上已经可以保持完全的负载平衡，因此DeepSeek-V3就不再使用token dropping的策略了。

## Multi-Token Prediction

Multi-Token Prediction（MTP），顾名思义，在前向计算的时候一步可以预测 >1 个token。

这样的多token预测策略可以在训练中使用，提升模型的远距离的理解能力；也可以用在推理中，加速inference输出，不过推理加速算是副产品了。