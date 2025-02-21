import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # 预计算频率参数 theta_i
        theta = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim)) # 对于hidden维度计算theta, 每一个维度, 变化的频率有差异.
        self.register_buffer('theta', theta)

    def forward(self, x):
        """
        x: (batch_size, seq_len, dim)
        返回旋转后的 x
        """
        batch_size, head, seq_len, dim = x.size()

        # 生成位置索引 m
        m = torch.arange(seq_len, device=x.device).float().unsqueeze(1)  # (seq_len, 1)

        # 计算旋转角度 m * theta_i
        freqs = m * self.theta.unsqueeze(0)  # (seq_len, dim // 2)

        # 将 freqs 扩展为复数形式 [cos(m*theta_i), sin(m*theta_i)]
        cos_freqs = torch.cos(freqs)  # (seq_len, dim // 2)
        sin_freqs = torch.sin(freqs)  # (seq_len, dim // 2)

        # 将 x 的每一对维度视为复数，进行旋转
        x_rotated = x.clone()
        x_rotated[..., 0::2] = x[..., 0::2] * cos_freqs - x[..., 1::2] * sin_freqs
        x_rotated[..., 1::2] = x[..., 0::2] * sin_freqs + x[..., 1::2] * cos_freqs

        return x_rotated


class SelfAttentionWithRoPE(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # 定义 Q, K, V 的线性变换
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # RoPE
        self.rope = RotaryPositionEmbedding(self.head_dim)

        # 输出线性变换
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        x: (batch_size, seq_len, dim)
        """
        batch_size, seq_len, dim = x.size()

        # 计算 Q, K, V
        q = self.q_proj(x)  # (batch_size, seq_len, dim)
        k = self.k_proj(x)  # (batch_size, seq_len, dim)
        v = self.v_proj(x)  # (batch_size, seq_len, dim)

        # 将 Q, K, V 拆分为多头
        # (batch_size, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 对 Q 和 K 应用 RoPE
        q = self.rope(q)
        k = self.rope(k)

        # 计算注意力分数
        # (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)

        # 计算加权和
        output = torch.matmul(attn_weights, v)  # (batch_size, num_heads, seq_len, head_dim)

        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)  # (batch_size, seq_len, dim)

        # 输出线性变换
        output = self.out_proj(output)

        return output


# 测试代码
if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    dim = 64
    num_heads = 8

    # 随机生成输入
    x = torch.randn(batch_size, seq_len, dim)

    # 创建 Self-Attention 模型
    model = SelfAttentionWithRoPE(dim, num_heads)

    # 前向传播
    output = model(x)
    print("输入形状:", x.shape)
    print("输出形状:", output.shape)
zhe