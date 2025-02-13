import torch

# 定义HMM参数（全部使用概率的log形式避免数值下溢）
states = ["晴天", "雨天"]
observations = ["散步", "购物", "打扫"]
n_states = len(states)
n_obs = len(observations)

# 初始概率（log形式）
pi = torch.log(torch.tensor([0.6, 0.4]))  # 初始更可能是晴天

# 转移概率矩阵（log形式）
A = torch.log(torch.tensor([
    [0.7, 0.3],  # 晴天 -> 晴天/雨天
    [0.4, 0.6]  # 雨天 -> 晴天/雨天
]))

# 发射概率矩阵（log形式）
B = torch.log(torch.tensor([
    [0.5, 0.3, 0.2],  # 晴天时的活动概率
    [0.1, 0.4, 0.5]  # 雨天时的活动概率
]))


def viterbi(obs_seq):
    """维特比算法实现"""
    T = len(obs_seq)

    # 初始化DP表格
    dp = torch.zeros((T, n_states)) - float('inf')
    path = torch.zeros((T, n_states), dtype=torch.long)

    # 初始化第一步
    dp[0] = pi + B[:, obs_seq[0]]

    # 递推计算
    for t in range(1, T):
        for s in range(n_states):
            # 找到到当前状态的最大概率路径
            prob, prev_state = torch.max(
                dp[t - 1] + A[:, s],
                dim=0
            )
            dp[t, s] = prob + B[s, obs_seq[t]]
            path[t, s] = prev_state

    # 回溯最佳路径
    best_path = torch.zeros(T, dtype=torch.long)
    best_path[-1] = torch.argmax(dp[-1])

    for t in range(T - 2, -1, -1):
        best_path[t] = path[t + 1, best_path[t + 1]]

    return [states[i] for i in best_path.tolist()]


# 示例：观察序列["散步", "打扫", "购物"]对应的索引为[0, 2, 1]
obs_seq = [0, 2, 1]
predicted_states = viterbi(obs_seq)

print("观察序列:", [observations[i] for i in obs_seq])
print("预测的天气序列:", predicted_states)