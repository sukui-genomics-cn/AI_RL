# MatMul Introduce
- torch.matmul(input, other, out=None) → Tensor
- 输入要求
  - 如果输入是2D张量(矩阵), 则执行标准的矩阵乘法
  - 如果输入是更高维度的张量, 则执行批量矩阵乘法

**矩阵乘法**
```python
a = torch.tensor([[1, 2], [3, 4]])  # 形状: (2, 2)
b = torch.tensor([[5, 6], [7, 8]])  # 形状: (2, 2)

# 矩阵乘法
result = torch.matmul(a, b)
print(result)
# 输出:
# tensor([[19, 22],
#         [43, 50]])


```

**批量矩阵乘法**
```python
a = torch.randn(3, 2, 4)  # 形状: (3, 2, 4)
b = torch.randn(3, 4, 5)  # 形状: (3, 4, 5)

# 批量矩阵乘法
result = torch.matmul(a, b)
print(result.shape)  # 输出: torch.Size([3, 2, 5])
```

**向量与矩阵的乘法**
```python
a = torch.tensor([1, 2, 3])  # 形状: (3,)
b = torch.tensor([[4, 5], [6, 7], [8, 9]])  # 形状: (3, 2)

# 向量与矩阵的乘法
result = torch.matmul(a, b)
print(result)  # 输出: tensor([40, 46])
```