import pickle
import numpy as np
import gymnasium as gym
from q_learning_agent import QLearningAgent


def train_q_agent(episodes=5000, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
    env = gym.make("CartPole-v1")
    agent = QLearningAgent()

    for episode in range(episodes):
        state, _ = env.reset()
        done = False

        while not done:
            state_disc = agent.discretize(state)

            # Epsilon-greedy policy
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(agent.q_table[state_disc]))

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_disc = agent.discretize(next_state)
            best_next = np.argmax(agent.q_table[next_disc])
            td_target = reward + gamma * agent.q_table[next_disc][best_next]
            agent.q_table[state_disc][action] += alpha * (td_target - agent.q_table[state_disc][action])

            state = next_state

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if (episode + 1) % 500 == 0:
            print(f"Episode {episode + 1}/{episodes}, epsilon: {epsilon:.3f}")

    with open("q_table.pkl", "wb") as f:
        pickle.dump(agent.q_table, f)

    env.close()
    print("Training complete. Q-table saved to q_table.pkl.")


if __name__ == "__main__":
    train_q_agent()