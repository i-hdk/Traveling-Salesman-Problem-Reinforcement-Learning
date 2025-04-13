import TSPEnv
import os

"""
type the following in console for a fresher run?:
%reset -f
%runfile C:/Users/isabe/Desktop/schl/cs/tsp/run.py --wdir
"""

env = TSPEnv.TSPEnv(num_cities=5)
obs, info = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    done = terminated or truncated

#os.system('cls' if os.name == 'nt' else 'clear')
