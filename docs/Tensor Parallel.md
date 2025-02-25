# Parallel

## Gather & Reduce

在torch的`torch.distributed` 模块中, `all_gather`和`all_reduce`是两种常用的分布式通信原语, 它们在多进程之间同步和聚合数据非常重要.
1. `all_gather`
   1. 作用: `all_gather` 用于将所有进程中的数据(通常是一个张量)收集起来, 并将结果分发给每个进程. 即每个进程都会收到所有进程的输入数据的完整副本.
   2. 过程: 
      1. 假设有N进程, 每个进程提供一个输入张量(可以是相同或不同的值)
      2. 调用`all_gather`后, 每个进程都会得到一个包含所有N个输入张量的列表或者张量组.
   3. 场景:
      1. 需要收集每个进程的计算结果时, 比如汇总所有设备的输出
      2. 分布式训练中, 可能需要收集每个进程的局部预测结果来做全局评估.
```python
import torch
import torch.distributed as dist

# 假设有4个进程
rank = dist.get_rank()
tensor = torch.tensor([rank]) # 每个进程输入自己的rank
output = [torch.zeros(1) for _ in range(4)] # 预分配空间
dist.all_gather(output, tensor)
print(output)  # 每个进程都会输出 [tensor(0), tensor(1), tensor(2), tensor(3)]
```

2. `all_reduce`:
   1. 作用: `all_reduce` 用于对所有进程中的数据(通常是一个张量)执行某种缩减操作 (如求和, 求平均, 取最大值), 并将结果广播给每个进程. 每个进程最终都会得到相同的缩减结果
   2. 过程:
      1. 每个进程提供一个输入张量
      2. 通过指定的操作(如sum, mean, max) 对所有输入张量进行聚合
      3. 聚合后的单一结果会被分发到所有进程
   3. 典型场景
      1. 在分布式训练中, 计算全局梯度平均值时常用. 如每个进程计算本地梯度后, 通过 `all_reduce` 得到全局梯度的平均值, 用于参数更新
```python
import torch
import torch.distributed as dist

# 假设有4个进程
rank = dist.get_rank()
tensor = torch.tensor([rank*1.0]) # 每个进程输入rank 的浮点值
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
print(tensor) # 每个进程都会输出 tensor(6.0), 因为0+1+2+3=6

```

## Modules

**NCCL**
NCCL是Nvidia专门为多GPU之间提供通讯的通讯库, 或者说是一个多GPU卡通讯的框架, 提供了包括AllReduce, Broadcast, Reduce, AllGather, ReduceScatter等集合通讯API, NCCL屏蔽了底层复杂的细节, 向上提供API提供训练框架调用, 向下连接机内间的GPU以完成模型参数的高效传输.

![img](https://developer.qcloudimg.com/http-save/yehe-9497423/b4197640cb6e5e9027e1ee76cbccc63a.png)

**Megatron-LM**
Megatron-LM是一个基于PyTorch的分布式训练框架, 用来训练基于Transformer的大模型. Megatron-LM综合应用了数据并行, 张量并行, 流水线并行(pipeline parallelism). 很多大模型的训练过程都采用它, 如bloom, opt等.

**troch.distributed (dist)**

为运行在一台或多台机器上的多个计算节点之间的Pytorch提供多进程并行性通信的原语. 其能轻松地并行化在跨进程和机器集群的计算.

Group是所有进程的子集.

Backend进程通信库, PyTorch支持NCCL, GLOO, MPI.

world_size 在进程组中的进程数

Rank分配给分布式进程组中每个进程的唯一标识符, 它们始终是从0到world_size的连续整数.

**troch.distributed算子介绍**

- gather: 把其他进程的数据收集到目标进程, 返回一个列表

![img](https://developer.qcloudimg.com/http-save/yehe-9497423/5b08254b2fa7839371eb131e9b1d4247.png)

- all_gather: 是将所有进程的数据收集起来, 再分发给他们

![img](https://developer.qcloudimg.com/http-save/yehe-9497423/433c4d4fa7dda49c38af9d2c01d0c061.png)

- reduce: 把所有节点的值加起来, 再分发给所有节点

![img](https://developer.qcloudimg.com/http-save/yehe-9497423/805032468fbcf6b2f944f0c180d34f54.png)

- broadcast: 把某个节点的数据分发给所有节点

![img](https://developer.qcloudimg.com/http-save/yehe-9497423/6eb890bf9406e2a685e504ec6c60a555.png)

- scatter: 把某个进程上的列表数据逐个分发给其他所有进程

![img](https://developer.qcloudimg.com/http-save/yehe-9497423/57db08d59cd22d1de8dc5d95149a30c3.png)



使用 `tf.autograph.experimental.do_not_convert` 装饰器允许在损失函数中设置断点.