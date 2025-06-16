import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

# === Environment Setup ===
# Create the MountainCar-v0 environment using Gymnasium.
env = gym.make("MountainCar-v0", render_mode=None)
# Get the lower and upper bounds for each dimension of the observation space.
obs_low = env.observation_space.low      # Minimum values for position and velocity
obs_high = env.observation_space.high    # Maximum values for position and velocity
# Get the number of possible discrete actions.
n_actions = env.action_space.n           # For MountainCar-v0, actions: 0=left, 1=neutral, 2=right

print(f"Observation space low: {obs_low}, high: {obs_high}")
print(f"Number of actions: {n_actions}")

"""
Observation space low: [-1.2  -0.07], high: [0.6  0.07]
Number of actions: 3
"""

# === Discretization ===
# The state space in MountainCar is continuous (position, velocity), so we discretize it
# into a finite number of bins for both position and velocity.
# bins: Number of discrete buckets for position and velocity, e.g., [18, 14]
bins = np.array([18, 14])
# Initialize the Q-table with zeros. Shape: (number of position bins, number of velocity bins, number of actions)
Q = np.zeros(tuple(bins) + (n_actions,))

# Discretization function: maps a continuous observation to a discrete state index.
def discretize(obs):
    # Calculate ratios of the observation within the observation space
    ratios = (obs - obs_low) / (obs_high - obs_low)
    # Clip ratios to avoid out-of-bounds due to floating point arithmetic
    ratios = np.clip(ratios, 0, 0.999)
    # Scale ratios to the number of bins and convert to integer indices
    return tuple((ratios * bins).astype(int))


# === Q-learning Hyperparameters ===
alpha = 0.1      # Learning rate: how much new information overrides old
gamma = 0.99     # Discount factor: how much future rewards are valued over immediate rewards
epsilon = 0.1    # Epsilon for ε-greedy policy: probability of choosing a random action (exploration)
episodes = 5000  # Number of training episodes

reward_log = []

# === Q-learning Training Loop ===
for ep in range(episodes):
    ''' Initialize environment at the start of each episode '''
    obs, _ = env.reset()
    print(f"Episode {ep+1} starting observation: {obs}")
    state = discretize(obs)  # Convert continuous observation to discrete state
    total_reward = 0
    done = False

    while not done:
        # ε-greedy policy: with probability epsilon, choose a random action (exploration),
        # otherwise choose the action with the highest Q-value for current state (exploitation)
        if np.random.rand() < epsilon:
            action = np.random.randint(n_actions)
        else:
            action = np.argmax(Q[state])

        # Step the environment using the chosen action.
        # Each step advances the physical simulation by ~0.02 seconds.
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_state = discretize(next_obs)  # Discretize the next observation
        done = terminated or truncated     # Episode ends if either terminated or truncated

        # Q-table update:
        # Q(s,a) ← Q(s,a) + α * [reward + γ * max_a' Q(s',a') - Q(s,a)]
        Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])
        state = next_state
        total_reward += reward

    reward_log.append(total_reward)
    if (ep + 1) % 500 == 0:
        print(f"Episode {ep+1}: Reward = {total_reward}")

# === Plot Training Curve ===
# Plot total reward per episode to visualize agent's learning progress
plt.plot(reward_log)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("MountainCar Q-Learning Training Curve")
plt.grid(True)
plt.show()

# === Save Q-table ===
# Save the learned Q-table to disk for later use in inference
with open("q_table.pkl", "wb") as f:
    pickle.dump(Q, f)
print("✅ Q 表已保存为 q_table.pkl")

# === Inference Phase: Use Q-table to Test the Agent (with rendering) ===
# Create a new environment instance with rendering enabled for visualization
env = gym.make("MountainCar-v0", render_mode="human")
obs, _ = env.reset()
state = discretize(obs)
done = False
total_steps = 0

while not done:
    # At each step, use the learned Q-table to select the best action (greedy policy)
    action = np.argmax(Q[state])
    # Step the environment using the chosen action
    # Each step advances the physical simulation by ~0.02 seconds.
    obs, reward, terminated, truncated, _ = env.step(action)
    state = discretize(obs)
    done = terminated or truncated
    total_steps += 1

print(f"🎯 推理完成，用 {total_steps} 步达到目标（或失败）")
env.close()