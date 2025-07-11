import gymnasium as gym
import numpy as np
from utils import *
import argparse

# Evaluation parameters
BENCHMARK_EPISODES = 10
MAX_SCORE = 500

def evaluate(agent_fn, device, render=False):
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
            state = torch_state(state, dtype=torch.float32).to(device)
            action = model.predict(state)

            state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            total_reward += reward

        rewards.append(total_reward)

    env.close()
    avg_reward = np.mean(rewards)
    print(f"Average score over {BENCHMARK_EPISODES} episodes: {avg_reward:.2f}")
    return avg_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained PyTorch model")
    parser.add_argument("-f", "--file", type=str, default="best_agent.pt", help="Path to the trained model file")
    
    args = parser.parse_args()
    file_path = args.file
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Evaluating {file_path} pytorch model:")
    model = torch.load(file_path, weights_only=False, map_location=device)
    evaluate(model, device)