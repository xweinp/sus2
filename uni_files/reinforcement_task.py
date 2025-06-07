import gymnasium as gym
import numpy as np

# Evaluation parameters
BENCHMARK_EPISODES = 10
MAX_SCORE = 500

def evaluate_agent(agent_fn, render=False):
    """
    Evaluates a trained agent by running it in CartPole-v1 environment.
    agent_fn: function returning an agent object with a .act(state) method.
    render: whether to render the environment.
    """
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    rewards = []

    for episode in range(BENCHMARK_EPISODES):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent_fn().act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        rewards.append(total_reward)

    env.close()
    avg_reward = np.mean(rewards)
    print(f"Average score over {BENCHMARK_EPISODES} episodes: {avg_reward:.2f}")
    return avg_reward