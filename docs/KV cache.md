# KV Cache

使用KV Cache的推理速度明显优于没有使用KV Cache的, 而生成的token越长提升越明显, 当最大生成token数为1000时, 近10倍的加速, 一次推理近6 min.

| Name               | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| 原理               | 避免重复计算, 讲需要重复计算的结果进行缓存, 需要缓存的的值为历史token对应的KV值 |
| 为什么只需要KV     | 预测新的token只与输入的最后一个token相关, 输入的最后一个token因为需要计算注意力分数, 而注意力的值需要将输入token的V值进行加权即得到结构, 进行加权就需要当前的Q与所有的K计算得到权重, 所以只需要缓存历史token的KV值 |
| 为什么存在重复计算 | 生成式模型每生成一个新token都需要调用整个模型进行一次推理, 历史token计算得到的中间激活值在Decoder架构模型中每次推理时都是一样的.<br />这是因为Decoder中, 当前token只用之前的token计算得到注意力, 通过Causal Mask实现. 即在推理时前面已经生成的字符不需要与后面的字符产生attention. |

![img](https://i-blog.csdnimg.cn/blog_migrate/3654034ca318f09e6689f9224ba50e74.gif#pic_center)

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/3b8cad24d1d8474b354b99a1568caf67.gif#pic_center)

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/e75dd90a81ad2b124f417614119a4687.png)

由于Causal Mask矩阵存在, 预测下一个token只与输入的最后一个token的QKV和历史token的KV有关. 

如果没有Causal Mask, 如Encoder, 每次推理时每个token需要考虑所有输入的token, 所以得到的注意力值都会变化. 

**Encoder中计算第i个token的atten值过程**, 其涉及所有token的K和V.

![img](https://img2024.cnblogs.com/blog/721540/202407/721540-20240702155158710-858174080.png)

**但是在Decoder中, 第i个token只能获取前i-1token的K, V来获取它的atten score**