## 概述
倒立摆问题是控制中的经典问题, 这个版本问题中, 钟摆以随机位置开始, 目标是将其向上摆动, 使其保持直立.

类型: 连续控制.

### 环境
**Observation & state**

![img.png](images/Pendulum-v1 observation.png)

State 是最原始的环境内部的表示, observation则是state的函数. 好比所看见的东西并不一定就是它们在世界中的真实状态, 而是经过大脑加工过的信息.
thetadot是摆杆的角速度, theta是摆杆的角度, 两者都是连续的取值范围为[-8, 8].

**Actions**

![img.png](images/Pendulum-v1 actions.png)

Action 是一个连续的动作空间, 代表的是摆杆的力矩, 取值范围为[-2, 2].

**Reward**

![img.png](images/Pendulum-v1 reward.png)

**初始状态**

从-pi和pi的随机角度, 以及-1和1之间的随机速度开始.

**Example**
```python
import gymnasium as gym

env = gym.make('Pendulum-v1').unwrapped

actor = Actor()
critic = Critic()
observation, _ = env.reset()  # 环境重置
action, action_logprob = actor.choose_action(observation)
observation_, reward, done, truncated, info = env.step(action)
```

reward的来源:  
1. 在强化学习中, 环境(如Pendulum-v1) 会根据智能体所采取的动作和当前状态, 计算出一个将奖励值
2. 这个奖励值反应了智能体的动作是否将环境状态朝着目标状态推进, 如Pendulum-v1中, 奖励函数的目标使杆子保持竖直, 减少角速度并节省控制能量.

`env.step(action)`的作用: 只能执行动作action后, 调用环境的step, 环境会:
1. 更新环境状态
2. 返回新的观测值`observation_`
3. 计算当前的即时奖励`reward`
4. 指出当前回合是否结束(通过`done/truncate`)
5. 可能会提供一些附加信息`info`
