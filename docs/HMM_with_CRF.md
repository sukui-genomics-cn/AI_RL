# HMM和CRF对比

> blog: https://www.cnblogs.com/liuxiaochong/p/14314547.html

HMM模型将标注看作马尔可夫链, 一阶马尔科夫链式针对相邻标注的关系进行建模, 其中每个标记对应一个概率函数. HMM是一种生成模型，定义了联合概率分布，其中x和y分别表示观察序列和相对应的标注序列的随机变量。为了能够定义这种联合概率分布，生成模型需要枚举出所有可能的观察序列，这在实际运算过程中很困难，因为我们需要将观察序列的元素看做是彼此孤立的个体即假设每个元素彼此独立，任何时刻的观察结果只依赖于该时刻的状态。HMM模型的这个假设前提在比较小的数据集上是合适的，但实际上在大量真实语料中观察序列更多的是以一种多重的交互特征形式表现，观察元素之间广泛存在长程相关性。在命名实体识别的任务中，由于实体本身结构所具有的复杂性，利用简单的特征函数往往无法涵盖所有的特性，这时HMM的假设前提使得它无法使用复杂特征 (它无法使用多于一个标记的特征）。

CRF模型的特点：首先，CRF在给定了观察序列的情况下，对整个的序列的联合概率有一个统一的指 数模型。一个比较吸引人的特性是其 损失函数 的凸面性。其次，条件随机域模型相比较改进的隐马尔可夫模型可以更好更多的利用待识别文本中所提供的上下文信息以得更好的实验结果。条件随机域在中文组块 识别方面有效，并避免了严格的独立性假设和数据归纳偏置问题。条件随机域(CRF)模型应用到了中文名实体识别中，并且根据中文的特点，定义了多种特征模 板。并且有测试结果表明：在采用相同特征集合的条件下，条件随机域模型较其他概率模型有更好的性能表现。再次，词性标注主要面临兼类词消歧以及未知词标注 的难题，传统隐马尔科夫方法不易融合新特征，而最大熵马尔科夫模型存在标注偏置等问题。CRFs具有很强的推理能力，并且能够使用复杂、有重叠性和非独立的特征进行训练和推理，能够充分地利用上下文信息作为特征，还可以任意地添加其他外部特征，使得模型能够获取的信息非常丰富。同时，CRFs解决了最大熵模型中的“label bias”问题。从理论上讲，CRFs非常适用于中文的词性标注。

**二者对比**

1. 全局最优or局部最优: HMM对转移概率和表现概率直接建模，统计共现概率，由于其只在局部做归一化，所以容易陷入局部最优； CRF是在全局范围内统计归一化的概率，是全局最优的解。
2. 拓扑结构: HMM和MEMM是一种有向图，CRF是一种无向图。
3. 生成式模型or判别式模型: CRF 是判别模型，对问题的**条件概率**分布建模，而 HMM 是生成模型，对**联合概率**分布建模。

> 联合概率: 描述多个事件同时发生的概率
>
> 条件概率: 描述在某一事件发生的条件下, 另一事件发生的概率.

## LSTM+CRF

**原理解析**
序列标注问题本质上是分类问题, 因为具有序列特征, 所以LSTM就很适合进行序列标注. 但这样的作法存在一个问题: **每个时刻的输出没有考虑上一时刻的输出**. 在利用LSTM进行序列建模时候只考虑输入序列的信息, 即单词信息, 但是没有考虑标签信息, 即输出标签信息. 这样会导致一个问题，以“我 喜欢 跑步”为例，LSTM输出“喜欢”的标签是“动词”，而“跑步”的标签可能也是“动词”。但是实际上，“名词”标签更为合适，因为“跑步”这里是一项运动。也就是“动词”+“名词”这个规则并没有被LSTM模型捕捉到。也就是说这样使用**LSTM无法对标签转移关系进行建模**. 而标签转移关系对序列标注任务来说是很重要的，所以就在LSTM的基础上引入一个**标签转移矩阵**对标签转移关系进行建模。**CRF有两类特征函数, 一类是针对观测序列与状态的对应关系(如"我"一般是"名词"), 一类是针对状态间关系(如"动词"后一般跟"名词"). 在LSTM+CRF中, 前一类特征函数的输出是由LSTM的输出替代, 后一类特征函数就变成了标签转移矩阵**.

如下图所示, 对于一个输入序列 $X=(x_1, x_2,x_3, x_4)$ , 经过Embedding后得到输入到LSTM中, 经过线性层作用后得到每个词对应到每个label上的分数. 这里label的集和包括起始标签S, 结束标签E, 以及一般标签L1, L2, L3.

![](https://picx.zhimg.com/v2-4a4a768bfe423fcebc4b9b78dc11418f_1440w.jpg)

> LSTM输出每个词在各个标签上的得分



![img](https://pic4.zhimg.com/v2-4774fb7b9b5c75b7e07edf5db2f38847_1440w.jpg)

> 标签转移矩阵T, 表示标签之间的转移得分.

一般来说, 对于一个序列x, 如果序列x的长度为n, 有m个可能的标签, 那么共有 $m^n$ 个可能的标记结构, 即 $m^n, y=(y_1,y_2,...,y_n)$ . 可以利用LSTM+CRF模型计算出每个可能的标记结构的得分$score(y)$, 然后利用softmax进行归一化求出某个标注结果的概率 $p(y|x)=e^{score(y)}/Z, Z=\sum_y{e^{score(y)}}$ , 选择概率最大的作为标注结果,.

## BiLSTM+CRF 介绍

> blog: https://www.cnblogs.com/zjuhaohaoxuexi/p/15257605.html

**前言**

对于命名实体识别任务，基于神经网络的方法非常普遍。例如，[Neural Architectures for Named Entity Recognition](https://arxiv.org/pdf/1603.01360.pdf)提出了一个使用word and character embeddings的BiLSTM-CRF命名实体识别模型。我将以本文中的模型为例来解释**CRF层是如何工作的**。

在条件随机场中, 通过选用指数函数并引入特征函数, 条件概率被定义如下
$$
P(Y|X) = \frac{1}{Z}exp(\sum\limits_j \sum\limits_{i=1}^{n-1} t_j(y_{i+1}, y_i, X, i) + \sum\limits_k \sum\limits_{i=1} s_k(y_i, X, i))
$$
这里的 $X$ 指的是整一个观测序列, 而且这里定义的条件概率计算方式, 就只是讲观测序列X作为条件, 并不对其作任何独立假设.

其中, $t_i(y_{i+1}, y_i, X, i)$ 是(发射函数)定义在观测序列的两个相邻标记位置上的转移特征函数, 用于刻相邻标记变量之间的相关关系以及观测序列对他们的影响. 即<u>给定观测序列X, 其标注序列在i 及 i-1 位置上的转移概率</u>! 而特征函数定义往往不止一种, 因此会有一个下标j代表要遍历计算每一种特征函数的取值.

$s_k(y_i, X, i)$ 是定义在观测序列的标记位置i上的<u>状态特征函数</u> , 用于刻画观测序列对标记变量的影响. 即表示对于观测序列X其i位置的标记概率. 同理, 也有多种特征函数, 所以会有下标k.

![img](https://upload-images.jianshu.io/upload_images/11525720-3250d903445c7828.png?imageMogr2/auto-orient/strip|imageView2/2/w/1050/format/webp) 

假设我们有一个数据集, 其中有两个实体类型: Person和Organization. 但事实上, 在我们的数据集中, 有5个实体标签:

B-Person, I-Person, B-Organization, I-Organization, O.

此外, x是一个包含5个单词的句子, w0, w1, w2, w3, w4. 更重要的是, 在句子x中, `[w0, w1]` 是一个Person实体, `[w3]` 是一个Organization实体, 其他都是 "O".

**模型介绍**

- 首先, 将句子x中的每个单词表示为一个向量, 其中包括单词的嵌入和字符的嵌入. 字符嵌入是随机初始化的, 词嵌入通过是一个预先训练的词嵌入文件导入的. 所有的嵌入将在训练过程中进行微调.
- BiLSTM-CRF模型的输入是这些嵌入, 输出是句子x中的单词的预测标签.

![image](https://img2020.cnblogs.com/blog/1852906/202109/1852906-20210912110931970-1291020015.png)

- 不用知道BiLSTM层的细节, 但为了更容易的理解CRF层, 我们需要知道BiLSTM输出的意义是什么.

![image](https://img2020.cnblogs.com/blog/1852906/202109/1852906-20210912111436851-1860855969.png)

BiLSTM层的输出是每一个标签的分数, 对于W0, BiLSTM节点的输出为1.5 (B-Person), 0.9 (I-Person), 0.1(I-Organization)和0.05(O), **这些分数将作为CRF层的输入.** 然后, **将BiLSTM层预测的所有分数输入CRF层. 在CRF层中, 选择预测得分最高的标签序列作为最佳答案.**

**为什么需要添加CRF层**

你可能已经发现，即使没有CRF层，也就是说，我们可以训练一个BiLSTM命名实体识别模型，如下图所示。
![image](https://img2020.cnblogs.com/blog/1852906/202109/1852906-20210912112148593-842090305.png)
因为每个单词的BiLSTM的输出是标签分数。我们可以选择每个单词得分最高的标签。例如，对于w0，“B-Person”得分最高(1.5)，因此我们可以选择“B-Person”作为其最佳预测标签。同样，我们可以为w1选择“I-Person”，为w2选择“O”，为w3选择“B-Organization”，为w4选择“O”。
虽然在这个例子中我们可以得到正确的句子x的标签，但是并不总是这样。再试一下下面图片中的例子。
![image](https://img2020.cnblogs.com/blog/1852906/202109/1852906-20210912112324356-452137530.png)

显然, 这次的输出是无效的, "I-Organization, I-Person"和"B-Organization, I-Person".

**CRF层可以向最终的预测标签添加一些约束, 以确保它们是有效的. 这些约束可以由CRF层在训练数据集中学习.** 

约束条件可以是:

- 句子中第一个单词标签应该以"B-" 或 "O"开头, 而不是"I-"
- “B-label1 I-label2 I-label3 I-…”，**在这个模式中，label1、label2、label3…应该是相同的命名实体标签**。例如，“B-Person I-Person”是有效的，但是“B-Person I-Organization”是无效的。
- “O I-label”无效。**一个命名实体的第一个标签应该以“B-”而不是“I-”开头**，换句话说，有效的模式应该是“O B-label”

有了这些有用的约束, 无效预测标签序列的数量将显著减少. 

### CRF Layer

第一个是emission分数. 这些emission分数来自BiLSTM层(word -> label的转移矩阵). 例如, 标记为B-Person的W0的分数为1.5.

![image](https://img2020.cnblogs.com/blog/1852906/202109/1852906-20210912121756702-1167395849.png)

每个标签对应的索引号如图:
![image](https://img2020.cnblogs.com/blog/1852906/202109/1852906-20210912121838514-1778501803.png)

我们用 $x_{iyj}$ 来表示emission分数。i是word的索引，yj是label的索引。如图2.1所示， $x_{i=1,yj=2}=x_{w1,B−Organization}=0.1$ ，即 w1作为B-Organization的得分为0.1。

**Transition Score**

我们使用 $t_{yi,yj}$ 来表示transition分数. 例如 $t_{B-Person, I-Person}=0.9$ 表示标签的transition, 即B-Person -> I-Person 转移得分为0.9. 因此, 我们有一个transition得分矩阵, 它存储了所有标签之间的所有得分.

为了使transition评分矩阵更健壮, 我们将添加另外两个标签, START和END. START是指一个句子的开头, 而不是第一个单词, END表示句子的结尾.

下面是一个transition得分矩阵的例子, 包括额外添加的START和END标签.

![image](https://img2020.cnblogs.com/blog/1852906/202109/1852906-20210912153658439-221709669.png)

如上表所示, 我们可以发现transition矩阵已经学习了一些有用的约束

- 句子中第一个单词的标签应该以"B-" 或 "O"开头, 而不是"I-"开头 (从"START"到"I-person或I-organization"的transition分数非常低)
- B-label1 I-label2 I-label3 I-…”,  在这个模式中,  label1, label2, label3...应该是相同的命名实体标签. 例如, “B-Person I-Person”是有效的，但是“B-Person I-Organization”是无效的。(例如，从“B--- Organization”到“I-Person”的分数只有0.0003，比其他分数低很多)
- “O I-label”无效。一个被命名实体的第一个标签应该以“B-”而不是“I-”开头，换句话说，有效的模式应该是“O B-label”(同样，的分数非常小)
- ...

你可能想问一个关于矩阵的问题, 在哪里或如何得到transition矩阵?

**实际上, 该矩阵是BiLSTM-CRF模型的一个参数, 在训练模型之前, 可以随机初始化矩阵中的所有transition分数. 所有的随机分数将在你的训练过程中自动更新. 换句话说, CRF层可以自己学习这些约束. **

### CRF Loss Function

CRF损失函数由真实路径得分和所有可能路径的总得分组成. 在所有可能的路径中, 真实路径的得分应该是最高的.

如, 我们的数据集中有如下表所示的这些标签:

![image](https://img2020.cnblogs.com/blog/1852906/202109/1852906-20210912155906841-1505882624.png)

我们还是有一个5个单词的句子。可能的路径是：

- 1. START B-Person B-Person B-Person B-Person B-Person END
- 1. START B-Person I-Person B-Person B-Person B-Person END
- …
- **10) START B-Person I-Person O B-Organization O END**
- …
- N) O O O O O O O

假设每条可能的路径都有一个分数 $P_i$ , 并且总共有N条可能的路径, 所有路径的总分数是
$$
P_{total} = P_1 + P_2 + ... + P_N = e^{S_1} + e^{S_2} + ... + e^{S_N}
$$
(在第2.4节中, 我们将解释如何计算 $S_i$ , 你也可以把它当作这条路径分数. )

如果我们说第10条路径是真正的路径, 换句话说, 第10条路径是我们的训练数据集提供的黄金标准标签. 在所有可能的路径中, 得分 $P_{10}$ 应该是百分比最大的.

**在训练过程中, CRF损失函数只需要两个分数: 真实路径的分数和所有可能路径的总分数. 所有可能路径的分数中, 真实路径分数所占的比例会逐渐增加.**

计算实际路径分数 $e^{S_i}$ 非常简单, 这里我们主要关注的是 $S_i$ 的计算. 

选取真实路径，**“START B-Person I-Person O B-Organization O END”**，我们以前用过，例如：

- 我们有一个5个单词的句子，w1,w2,w3,w4, w4,w5
- 我们增加了两个额外的单词来表示一个句子的开始和结束，w0,w6
- $S_i$ 由两部分组成: $S_i = EmissionScore + TransitionScore$

**Emission得分**
$$
EmissionScore = x_{0,Start} + x_{1, B-Person} + x_{2, I-Person} + x_{3,O} + x_{4, B-Organization} + x_{5, O} + x_{6,END}
$$

- $x_{index, label}$ 是第index个单词被label标记的分数.
- 这些得分 $x_{1,B-Person}, x_{2, I-Person}, x_{3,O}, x_{4, Organization}, x_{5, O}$ 来自之前的BiLSTM输出.
- 对于 $x_{0,START}, x_{6,END}$ , 我们可以把它设为0.

**Transition得分**
![image](https://img2020.cnblogs.com/blog/1852906/202109/1852906-20210912165045137-1306523038.png)

- $t_{label1->label2}$ 是从label1到label2的transition分数
- **这些分数来自CRF层, 换句话说, 这些transition分数实际上是CRF层的参数**

**所有可能的路径的总得分**
**衡量总分最简单的方法是: 列举所有可能的路径并将它们的分数相加. 但这里训练时间非常长** 

**步骤1: CRF Loss Function**
$$
L= \frac{P_1+P_2+...+P_N}{P_{realPath}} \\

LogL = -(log(e^{S_{RealPath}}) - log(e^{s_1}+e^{s_2}+...+e^{s_N}))
$$
**模型定义**

```python
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag2ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tag2ix = tag2ix
        self.tagset_size = len(tag2ix)
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim//2, num_layers=1, bidirectional=True)
        
        # maps output of lstm to tog space
        # 即每个单词产生的Emission(发射) Score分布包含产生"<s>", <e>的概率
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        
        # matrix of transition parameters
        # entry i, j is the score of transitioning to i from j
        # 所以, transition矩阵的每一列的元素之和为1, tag间的转移矩阵, 是CRF层的参数
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        
        # these two statements enforce the constraint that we never transfer to the 
        # start tag and we never transfer from the stop tag.
        self.transitions.data[tag2ix[START_TAG], :] = -10000
        self.transitions.data[:, tag2ix[END_TAG]] = -10000
        
    # 作为lstm刚开始输入时的初值(h_0,c_0)，为一个元组
    # 初值的维数为num_layers * num_directions, batch, hidden_size
    # 说明一批中训练计算时用的是一样的h_0,c_0，不同批次使用的是不同的h_0,c_0
    def init_hidden(self):
        return (torch.randn(2, 1,self.hidden_dim//2),
                torch.randn(2, 1,self.hidden_dim//2))
        
    # 求log(e^s1 + e^s2 + e^s3 + ... + e^sN)的值，应用动态归化算法
    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -1000)
        # START_TAG has all of the score
        init_alphas[0][self.tag2ix[START_TAG]] = 0
        # Wrap in a variable so that we will get automatic backprob
        forward_var = init_alphas
        # Interate through the sentence
        for feat in feats:
            # feat 指Bi-LSTM模型每一步的输出, 大小为tagset_size
            alphas_t = [] # the forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regradless of the previouse step
                emit_score = feat[next_tag].view(1,-1).expand(1,self.tagset_size)
                # the i_th entry of trans_score is the score of transitioning to next_tag from i
                trans_score = self.transitions[next_tag].view(1,-1)
                # The ith entry of next_tag_var is the value for the edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the score.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1,-1) # Need to Know
        terminal_var = forward_var + self.transitions[self.tag2ix[END_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha
    
    def _score_sentence(self, feats, tags):
        # gives the score of a provides tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag2ix[START_TAG]], dtype=torch.long),tags])
        # feats 中的i位置对应tags中的i+1位置
        # 这里的实现版本的emisson score为句子所有的真实单词位置的, 不包含认为添加的START_TAG位置
        # 和END_TAG位置, 而transition score则包含START_TAG微信hi -> 句子的第一个真实单词之间
        # 的转换分数和句子的最后一个真实单词 -> END_TAG位置之间的转换分数
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag2ix[END_TAG], tags[-1]]
        return score
        
    def _viterbi_decode(self, feats):
        backpointers = []
        # Initialize the viterbi variables in log space
        init_vars = torch.full((1, self.tagset_size), -10000.)
        init_vars[0][self.tag2ix[START_TAG]] = 0
        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vars
        for feat in feats:
            bptrs_t = [] # hold the back pointers for this step
            viterbivars_t = [] # holds the viterbi variables for this step
            
            for next_tag in range(self.tagset_size):
                #　next_tag_var[i] holds the viterbi variable for tag i 
                # at the previous step, plus the score of transitioning from tag i to
                # next_tag. We dont include the emission scores here because max does
                # not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set of 
            # viterbi variables we just computed
            forward_var - (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)
            
        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag2ix[END_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]
        
        # Follow the back pointer to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag2ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path
    
    def neg_log_likelihood(self, sentence, tags):
        # 由LSTM层计算得得每一时刻属于某一tag得值
        feats = self._get_lstm_features(sentence)
        # log(e^s1 + e^s2 + e^s3 + ... + e^sN) -> 仅前向传播, 获取预测得最优路径
        forward_score = self._forward_alg(feats)
        # 正确路径得值, s_realPath
        gold_score = self._score_sentence(feats, tags)
        # -(s_realpath - log(e^s1 + e^s2 + e^s3 + ... + e^sN)）
        return forward_score - gold_score
        
```

中带你关注一下`_forward_alg` 这个函数, 就是我们上面讲的求解 $log(e^{s_1} + e^{s_2} + e^{s_3} + ... + e^{s_N})$ , 用动态规划时的递推公式原理写出来, 以利于理解.

使用动态规划的思想, 逐个时间步的计算, 在每个时间步t, 我们维护一个长度为tag_size的向量 $cur^{t}$ . 这里只是未来论述的方便.

下式中, $S_{N_j}^t$ 为从序列开始到此时间步t的所有路径中时间步t标签为j的路径的各自得分.

![image](https://img2020.cnblogs.com/blog/1852906/202109/1852906-20210915224357316-550414631.jpg)

![image](https://img2020.cnblogs.com/blog/1852906/202109/1852906-20210915224706417-317017160.jpg)

**Main Function**

```python
if __name__ == "__main__":
    START_TAG = "<s>"
    END_TAG = "<e>"
    EMBEDDING_DIM = 5
    HIDDEN_DIM = 4

    # Make up some training data
    # 注意：这里的两个样本的长度不一样长
    training_data = [(
        "the wall street journal reported today that apple corporation made money".split(),
        "B I I I O O O B I O O".split()
    ), (
        "georgia tech is a university in georgia".split(),
        "B I O O O O B".split()
    )]
    # 构造单词->单词序号的映射字典(这个字典中不包括认为的句子开始单词: "<START>")
    # 以及认为的句子结束单词："<STOP>")
    # 构造单词->单词序号的映射字典(这个字典中不包括人为的句子开始单词："<START>"，以及人为的句子结束单词："<STOP>")
    word2ix = {}
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word2ix:
                word2ix[word] = len(word2ix)

    tag2ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, END_TAG: 4}
    model = BiLSTM(len(word2ix), tag2ix, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    
    # Check predictions before training
    # 输出训练前的预测序列
    with torch.no_grad():
        precheck_sent = prepare_sequence(training_data[0][0], word2ix)
        precheck_tags = torch.tensor([tag2ix[t] 
                                      for t in training_data[0][1]], dtype=torch.long)
        print(model(precheck_sent))
        
        # Make sure prepare_sequence from earlier in the LSTM section is loaded
        for epoch in range(300):
            for sentence, tags in training_data:
                # Step1. Remember that torch accumulates gradients.
                # we need to clear them out before each instance
                model.zero_grad()
                
                # Step2. Get our inputs ready for the network, that is 
                # turn them into Tensors of word indices.
                sentence_in = prepare_sequence(sentence, word2ix)
                # 训练时, 句子的tag序列也没有经过额外的处理(认为增加句子开始, 结束tag)
                targets = torch.tensor([tag2ix[t] for t in tags], dtype=torch.long)
                
                # Step3. Run our forward pass.
                loss = model.neg_log_likelihood(sentence_in, targets)
                
                # step4. Compute the loss, gradients, and update the parameters by
                # calling optimizer.step()
                loss.backward()
                optimizer.step()
    
    # Check predictions after training
    with torch.no_grad():
        precheck_sent = prepare_sequence(training_data[0][0], word2ix)
        print(model(precheck_sent))

    # 输出结果
    # (tensor(-9996.9365), [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    # (tensor(-9973.2725), [0, 1, 1, 1, 2, 2, 2, 0, 1, 2, 2])
    
```

**Refs**
[机器学习基础（11）条件随机场的理解及BI-LSTM+CRF实战](https://www.jianshu.com/p/45c5be21daa6)
[CRF Layer on the Top of BiLSTM](https://createmomo.github.io/2017/09/12/CRF_Layer_on_the_Top_of_BiLSTM_1/)