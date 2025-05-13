import TSPEnv
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import random

#runs while training to keep track of training over time
class RewardTrackerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_rewards = 0

    def _on_step(self) -> bool:
        # Accumulate reward
        self.current_rewards += self.locals["rewards"][0]

        # Check if episode is done
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_rewards)
            self.current_rewards = 0
        return True

    def plot_rewards(self):
        plt.figure(figsize=(10, 4))
        plt.plot(self.episode_rewards, label="Episode Reward")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Training Reward Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()

#TODO add greedy policy

def evaluate_random_policy(env, episodes=3):
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_distance = 0

        print(f"\n[Random Policy] Episode {ep + 1}")
        while not done:
            #action = env.action_space.sample() #FIXME can only sample non visited cities
            action = 0
            visited = env.get_visited()
            to_visit = visited.size-visited.sum()
            chosen = random.randint(1,to_visit)
            x = 0
            j = 0
            for i in visited:
                if i == 0:
                    x+=1
                if x == chosen:
                    action = j
                    break
                j+=1
                
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            done = terminated or truncated
        print(f"Finished tour. Total Distance: {info['total_distance']:.2f}")

def evaluate_trained_agent(env, model, episodes=3):
    for ep in range(episodes):
        obs, info = env.reset()
        done = False

        print(f"\n[Trained Agent] Episode {ep + 1}")
        while not done:
            action, _ = model.predict(obs, deterministic=True) #deterministic=true: always pick best option
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            done = terminated or truncated
        print(f"Finished tour. Total Distance: {info['total_distance']:.2f}")


env = TSPEnv.TSPEnv(num_cities=5)
check_env(env)
model = PPO("MultiInputPolicy", env, verbose=1) #TODO try different policies or models
callback = RewardTrackerCallback()
model.learn(total_timesteps=10000 * 4, callback=callback) #first num is episodes
callback.plot_rewards()

#TODO add save/load model with a separate file to separate training & testing 

evaluate_random_policy(env)
evaluate_trained_agent(env, model)

#TODO add bar graphs with random / greedy / different trained models on avg total dist travelled

env.close()

