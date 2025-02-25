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