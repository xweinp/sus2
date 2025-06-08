# How to use

To run the code, first create a virtual environment.

You need to make the script executable first:

```bash
chmod +x setup.sh
```

To create a virtual environment, train a model and evaluate this mode and best model run:
```bash
./setup.sh
```

Now you can also activate the virtual environment with:

```bash
source pysus/bin/activate
```

Scripts `train_agent.py` and `evaluate.py` come with many options you can set. To see them run:

```bash
python3 train_agent.py -h
python3 evaluate.py -h
```