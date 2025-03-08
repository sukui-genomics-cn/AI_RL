# Mamba

**Mamba前身S4的4个参数不随输入不同而不同**
![img](https://i-blog.csdnimg.cn/blog_migrate/2bc09dd2ed57895a3daf1a9172ac6008.png)

## 简化的SSM架构及最终的整体流程

将大多数SSM架构(H3)的基础模块, 与现代神经网络如(Transformer)中普遍存在的Gate MLP 想结合, 组合成新的Mamba块, 然后重复这个块(且与归一化和残差连接结合), 便构成了Mamba块.

![img](https://i-blog.csdnimg.cn/blog_migrate/a1b48eaa302739fd310ba21fc86da277.png)

- 为何要做线性投影: 经过线性投影后, 输入嵌入的维度可能会增加, 以便让模型能够处理更高维度的特征空间, 从而捕获更细致, 更复杂的特征
- 为什么SSM前有卷积: 
  - SSM之前的CNN负责提取局部特征(因其擅长捕捉局部的短距离特征), 而SSM则负责处理这些特征并捕捉序列数据中的长期依赖, 两者算互相补充.
  - CNN有助于建立token之间的局部上下文依赖, 从而防止独立的token计算. 比较若每个token独立计算, 那么模型就会丢失序列中token之间的上下文信息. 通过先进行卷积操作, 可以确保在进入ssm之前, 序列中的每个token已经考虑其
  邻居token的信息, 这样模型就不会单独地处理每个token, 而是在处理时考虑了整个局部上下文.



**Mamba 块的模型架构**

![img](https://i-blog.csdnimg.cn/blog_migrate/25c8d3157c954ca05b2da1374cea17bc.png)

其中的"选择性SSM"具有一下属性:

![img](https://i-blog.csdnimg.cn/blog_migrate/f35348495fe1790d40ed594ce11773d6.png)

1. Recurrent SSM通过离散化创建循环SSM
2. HiPPO对矩阵A进行初始化以捕获长距离依赖
3. 选择性扫描算法(Selective scan algorithm)选择性压缩信息
4. 硬件感知算法(Hardware-aware algorithm)用于加速计算
