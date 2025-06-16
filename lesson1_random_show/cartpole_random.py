import gymnasium as gym
import numpy as np

# 创建环境
env = gym.make("MountainCar-v0")
obs_low = env.observation_space.low
obs_high = env.observation_space.high
n_actions = env.action_space.n

# 离散化参数
bins = np.array([18, 14])  # position 分18份，velocity 分14份

# 创建 Q 表
Q = np.zeros(tuple(bins) + (n_actions,))

# 离散化函数
def discretize(obs):
    ratios = (obs - obs_low) / (obs_high - obs_low)
    ratios = np.clip(ratios, 0, 0.999)
    return tuple((ratios * bins).astype(int))

# 超参数
alpha = 0.1       # 学习率
gamma = 0.99      # 折扣率
epsilon = 0.1     # ε-greedy
episodes = 5000   # 总训练轮数

# 记录每集奖励
reward_log = []

for ep in range(episodes):
    obs, _ = env.reset()
    state = discretize(obs)
    total_reward = 0
    done = False

    while not done:
        # ε-greedy 策略
        if np.random.rand() < epsilon:
            action = np.random.randint(n_actions)
        else:
            action = np.argmax(Q[state])

        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_state = discretize(next_obs)
        done = terminated or truncated

        # Q 表更新
        Q[state][action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state][action]
        )

        state = next_state
        total_reward += reward

    reward_log.append(total_reward)
    if (ep + 1) % 500 == 0:
        print(f"Episode {ep+1}, total reward: {total_reward}")

env.close()