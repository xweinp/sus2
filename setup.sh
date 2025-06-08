#!/bin/bash

echo "Setting up the Python environment..."
python3 -m venv pysus

source pysus/bin/activate

echo "Installing required packages..."
pip install -r requirements.txt

# To activate the virtual environment, run:
# source pysus/bin/activate

echo "Evaluating best agent..."
python3 evaluate.py -f best_agent.pt

echo "Setup complete. Run the training script."
python3 train_agent.py

echo "Evaluating the agent we have just trained..."
python3 evaluate.py -f agent.pt

