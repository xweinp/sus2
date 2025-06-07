class MyRLAgent:
    def __init__(self):
        pass

    def act(self, state):
        # Always takes action 0, regardless of state.
        return 0

if __name__ == "__main__":
    from reinforcement_task import evaluate_agent
    print("Evaluating a very weak agent (always picks action 0):")
    evaluate_agent(lambda: MyRLAgent())