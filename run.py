import TSPEnv
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import random

#TODO add greedy policy
#def evaluate_greedy_policy

def evaluate_random_policy(env, episodes=3):
    random_avg_total_dist = 0
    for ep in range(episodes):
        obs, info = env.reset(seed=ep)
        done = False
        total_distance = 0

        print(f"\n[Random Policy] Episode {ep + 1}")
        while not done:
            #action = env.action_space.sample()
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
        random_avg_total_dist+=info['total_distance']
    random_avg_total_dist/=episodes
    return random_avg_total_dist

def evaluate_trained_agent(env, model, episodes=3):
    trained_avg_total_dist = 0
    for ep in range(episodes):
        obs, info = env.reset(seed=ep)
        done = False

        print(f"\n[Trained Agent] Episode {ep + 1}")
        while not done:
            action, _ = model.predict(obs, deterministic=True) #deterministic=true: always pick best option
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            done = terminated or truncated
        print(f"Finished tour. Total Distance: {info['total_distance']:.2f}")
        trained_avg_total_dist+=info['total_distance']
    trained_avg_total_dist/=episodes
    return trained_avg_total_dist

print("running")
env = TSPEnv.TSPEnv(num_cities=5)
#check_env(env)
model = PPO.load("PPO_model")

#TODO add save/load model with a separate file to separate training & testing 

random_avg_total_dist = evaluate_random_policy(env,episodes=10)
trained_avg_total_dist = evaluate_trained_agent(env, model,episodes=10)

#TODO add bar graphs with random / greedy / different trained models on avg total dist travelled
print(f"random policy average total distance: {random_avg_total_dist:.3f}")
print(f"trained agent average total distance: {trained_avg_total_dist:.3f}")
plt.figure()
categories = ["Random Policy", "Trained Agent"]
values = [random_avg_total_dist, trained_avg_total_dist]
plt.bar(categories, values)
#plt.xlabel("Categories")
plt.ylabel("Total Distance Average")
plt.title("Total Distance Averages for Different Policies")
plt.show(block=False)

env.close()

input("Press Enter to exit...")


