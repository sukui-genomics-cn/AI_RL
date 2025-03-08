import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt


# 定义稀疏自编码器模型
class SparseAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, sparsity_param, beta):
        super(SparseAutoencoder, self).__init__()

        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)
        self.sparsity_param = sparsity_param  # 稀疏性参数
        self.beta = beta  # 稀疏性损失权重

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))  # 编码过程
        decoded = torch.sigmoid(self.decoder(encoded))  # 解码过程
        return decoded, encoded

    def sparsity_loss(self, p_h):
        # 计算稀疏性损失
        return self.beta * torch.sum(self.sparsity_param * torch.log(self.sparsity_param / p_h) +
                                     (1 - self.sparsity_param) * torch.log((1 - self.sparsity_param) / (1 - p_h)))


# 生成模拟遥感图像数据 (64 维特征)
X = np.random.rand(1000, 64)  # 1000 个样本，每个样本有 64 维光谱特征
X = torch.tensor(X, dtype=torch.float32)

# 创建数据加载器
dataset = TensorDataset(X, X)  # 输入和目标均为原始数据
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 定义模型、优化器
input_size = 64
hidden_size = 8  # 隐藏层大小
sparsity_param = 0.05  # 稀疏性参数
beta = 1  # 稀疏性损失权重
sae = SparseAutoencoder(input_size=input_size, hidden_size=hidden_size, sparsity_param=sparsity_param, beta=beta)
optimizer = optim.Adam(sae.parameters(), lr=0.001)

# 训练 SAE 模型
num_epochs = 50
for epoch in range(num_epochs):
    for data in dataloader:
        optimizer.zero_grad()
        inputs, original_data = data
        reconstructed_data, encoded_data = sae(inputs)  # 前向传播
        loss = nn.functional.binary_cross_entropy(reconstructed_data, original_data)  # 重构损失
        # 计算稀疏性损失
        p_h = torch.mean(encoded_data, dim=0)
        # sparsity_loss = 0.01 * torch.mean(torch.abs(encoded_data))  # L1稀疏惩罚
        loss += sae.sparsity_loss(p_h)  # 加入稀疏性损失
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# 使用训练好的模型进行特征提取
with torch.no_grad():
    _, encoded_data = sae(X)  # 提取编码后的特征
    encoded_data = encoded_data.numpy()

# 可视化原始数据和编码后的特征
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Original Data')
plt.imshow(X.numpy()[:10], aspect='auto', cmap='hot')

plt.subplot(1, 2, 2)
plt.title('Encoded Features')
plt.imshow(encoded_data[:10], aspect='auto', cmap='hot')

plt.show()
