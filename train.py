import TSPEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

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
        plt.show(block=False)

print("training model...")
env = TSPEnv.TSPEnv(num_cities=15)
check_env(env)
model = PPO("MultiInputPolicy", env, verbose=1) #TODO try different policies or models
callback = RewardTrackerCallback()
model.learn(total_timesteps=10000 * 4, callback=callback) #first num is episodes
callback.plot_rewards()
model.save("PPO_model")

input("Press Enter to exit...")
