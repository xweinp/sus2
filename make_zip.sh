#!/bin/bash


echo "Creating zip for submission..."
zip -r jp459481_SUS-task2.zip \
    utils.py \
    train_agent.py \
    evaluate.py \
    requirements.txt \
    best_agent.pt \
    README.md \
    setup.sh \
    report.pdf