import logging

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# 定义策略网络（Actor）和价值网络（Critic）
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        action_probs = torch.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value

# PPO算法
class PPO:
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=4):
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.policy_old = ActorCritic(state_dim, action_dim, hidden_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())

    def select_action(self, state):
        print("State shape:", np.shape(state))  # 添加此行以检查 state 的形状
        state = torch.FloatTensor(state)
        action_probs, _ = self.policy_old(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, memory):
        # 将存储的数据转换为张量
        states = torch.FloatTensor(memory['states'])
        actions = torch.LongTensor(memory['actions'])
        rewards = torch.FloatTensor(memory['rewards'])
        old_log_probs = torch.FloatTensor(memory['log_probs'])

        # 计算折扣回报
        returns = []
        discounted_sum = 0
        for reward in reversed(rewards):
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        returns = torch.FloatTensor(returns)

        # 计算优势
        _, state_values = self.policy(states)
        advantages = returns - state_values.detach().squeeze()

        # 更新策略
        for _ in range(self.K_epochs):
            action_probs, state_values = self.policy(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)

            # 计算概率比率
            ratios = torch.exp(new_log_probs - old_log_probs)

            # 计算PPO的损失函数
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2).mean() + 0.5 * nn.MSELoss()(state_values.squeeze(), returns)

            # 反向传播并更新参数
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # 更新旧的策略
        self.policy_old.load_state_dict(self.policy.state_dict())

# 训练PPO算法
def train_ppo(env_name='CartPole-v1', max_episodes=1000, max_timesteps=300):
    logging.info(f'Training PPO on {env_name}')
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    ppo = PPO(state_dim, action_dim)
    memory = {'states': [], 'actions': [], 'rewards': [], 'log_probs': []}

    for episode in range(max_episodes):
        state = env.reset()
        for t in range(max_timesteps):
            action, log_prob = ppo.select_action(state)
            next_state, reward, done, _ = env.step(action)

            # 存储数据到memory
            memory['states'].append(state)
            memory['actions'].append(action)
            memory['rewards'].append(reward)
            memory['log_probs'].append(log_prob.item())

            state = next_state

            if done:
                break

        # 更新策略
        ppo.update(memory)

        # 清空memory
        memory = {'states': [], 'actions': [], 'rewards': [], 'log_probs': []}

        # 打印进度
        if (episode + 1) % 10 == 0:
            print(f'Episode {episode + 1}/{max_episodes} completed')

    env.close()

if __name__ == '__main__':
    train_ppo()
