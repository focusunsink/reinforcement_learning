import gymnasium as gym
import time

env = gym.make("MountainCar-v0", render_mode="human")
obs, info = env.reset()
done = False

for _ in range(1000):
    action = env.action_space.sample()  # 随机动作：0,1,2
    if action == 0:
        print("push left")
    elif action == 1:
        print("do nothing")
    else:
        print("push right")    
    
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Observation: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
    env.render()  
    time.sleep(0.02)
    if terminated or truncated:

        print("Resetting...")
        obs, info = env.reset()