import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os


class TSPEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, num_cities=5):
        super().__init__()

        self.num_cities = num_cities

        # Action: pick next city (Discrete)
        self.action_space = spaces.Discrete(num_cities)

        # Observation: visited cities and current city
        self.observation_space = spaces.Dict({
            "visited": spaces.MultiBinary(num_cities),
            "current_city": spaces.Discrete(num_cities)
        })

        self.seed()
        self.cities = None
        self.visited = None
        self.current_city = None
        self.total_distance = None
        self.steps = None

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.seed(seed)

        self.cities = self.np_random.random((self.num_cities, 2))  # Tuple shape
        self.visited = np.zeros(self.num_cities, dtype=np.int8)
        self.current_city = self.np_random.integers(self.num_cities)
        self.visited[self.current_city] = 1

        self.total_distance = 0.0
        self.steps = 0

        obs = {"visited": self.visited.copy(), "current_city": self.current_city}
        info = {}
        return obs, info

    def step(self, action):
        if self.visited[action]:
            # Penalize if visiting already visited city
            reward = -10.0
            terminated = False
        else:
            distance = np.linalg.norm(self.cities[self.current_city] - self.cities[action])
            self.total_distance += distance
            reward = -distance
            self.current_city = action
            self.visited[action] = 1
            self.steps += 1
            terminated = bool(np.all(self.visited))



        truncated = False
        obs = {"visited": self.visited.copy(), "current_city": self.current_city}
        info = {"total_distance": self.total_distance}
        return obs, reward, terminated, truncated, info

    def render(self):
        #os.system('cls' if os.name == 'nt' else 'clear')

        print(f"Step: {self.steps}")
        print(f"Visited: {self.visited}")
        print(f"Current City: {self.current_city}")
        print(f"Total Distance: {self.total_distance:.2f}")
