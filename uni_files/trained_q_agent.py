import pickle
from q_learning_agent import QLearningAgent
from reinforcement_task import evaluate_agent

class TrainedQLearningAgent:
    def __init__(self):
        with open("q_table.pkl", "rb") as f:
            q_table = pickle.load(f)
        self.agent = QLearningAgent(q_table=q_table)

    def act(self, state):
        return self.agent.act(state)

if __name__ == "__main__":
    print("Evaluating the trained Q-learning agent:")
    evaluate_agent(lambda: TrainedQLearningAgent())