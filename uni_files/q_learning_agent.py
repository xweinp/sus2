import gymnasium as gym
import numpy as np

# Discretization bins for each observation space dimension
NUM_BINS = (6, 12, 6, 12)

class QLearningAgent:
    def __init__(self, bins=NUM_BINS, q_table=None):
        self.bins = bins
        self.env = gym.make("CartPole-v1")
        self.obs_low = self.env.observation_space.low
        self.obs_high = self.env.observation_space.high

        # Manually set limits for velocity dimensions (which are infinite)
        self.obs_low[1] = -3.0
        self.obs_high[1] = 3.0
        self.obs_low[3] = -3.0
        self.obs_high[3] = 3.0

        self.q_table = q_table if q_table is not None else np.zeros(self.bins + (self.env.action_space.n,))

    def discretize(self, obs):
        ratios = [(obs[i] - self.obs_low[i]) / (self.obs_high[i] - self.obs_low[i]) for i in range(len(obs))]
        new_obs = [int(round((self.bins[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.bins[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

    def act(self, state):
        state_disc = self.discretize(state)
        return int(np.argmax(self.q_table[state_disc]))