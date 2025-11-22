# Lab 8: Markov Decision Process and Dynamic Programming

Solutions for Lab Assignment 8 covering MDP and Reinforcement Learning.

## Problems

### Problem 1: Grid World Value Iteration ([lab8_1.py](lab8_1.py))
4×3 stochastic grid world with Value Iteration algorithm. Solved for 5 different step costs.

### Problem 2: Gbike Rental - Original ([lab8_2.py](lab8_2.py))
Policy Iteration for bicycle rental inventory management with Poisson-distributed demands.

### Problem 3: Gbike Rental - Modified ([lab8_3.py](lab8_3.py))
Extended version with free employee shuttle and parking overflow costs.

## Running

```bash
# Activate virtual environment
source .venv/bin/activate

# Run individual problems
python lab8_1.py
python lab8_2.py
python lab8_3.py
```

## Visualizations

All outputs are saved in the `visualizations/` directory.

## Dependencies

- numpy
- matplotlib
- seaborn
