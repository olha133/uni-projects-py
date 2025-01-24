# 8 Puzzle Solver Using Q-Learning

This project solves the 8-puzzle problem, a classic AI challenge, using the Q-learning reinforcement learning algorithm. The aim is to rearrange a 3x3 grid of tiles numbered 1 to 8, with one blank space, to match the target configuration. The implementation uses custom Gym environments and reinforcement learning techniques.

---

## Problem Description

The **8-puzzle problem** is a sliding puzzle where the objective is to move tiles to achieve the target configuration by utilizing the blank space. 

### State Space
The state space consists of all possible configurations of a 3x3 grid, leading to approximately `9!` states (362,880 states).

### Actions
The possible actions correspond to moving the blank space in four directions:
- **0**: Move the blank space **up**.
- **1**: Move the blank space **down**.
- **2**: Move the blank space **left**.
- **3**: Move the blank space **right**.

### Goal
The problem is considered solved when the tiles match the target configuration.

---

## Reward Function

The reward function incentivizes the agent to solve the puzzle efficiently:
1. **-1** for each move.
2. **+1** for each tile placed in a new correct position.
3. **+20** when the puzzle is solved.

---

## Q-Learning Implementation

Q-learning is a model-free reinforcement learning algorithm that learns the optimal policy for an agent by maximizing cumulative rewards.

### Key Concepts
- **Q-value**: Represents the expected cumulative reward for taking an action in a given state.
- **Exploration vs Exploitation**: Balances trying new actions (exploration) with choosing the best-known actions (exploitation) using an ε-greedy strategy.
- **Update Rule**: Q-values are updated iteratively using the Bellman equation.

### Implementation Highlights
The `QLearningAgent` class implements Q-learning with:
1. **State Representation**: States are represented as 3x3 grids.
2. **Q-table**: A table mapping states and actions to Q-values.
3. **Training**: The agent trains over multiple episodes, updating Q-values and reducing exploration (ε) over time.

---

## Experiments

### Metrics
1. **Training Time**: Time taken for a specific number of episodes.
2. **Convergence Rate**: How quickly the agent learns the optimal policy.
3. **Success Rate**: Percentage of episodes where the agent solves the puzzle.
4. **Stability**: Average number of steps to solve the puzzle from different initial states.

### Results

| Experiment | Alpha | Gamma | Epsilon | Epsilon Decay | Min Epsilon | Steps     | Time (s) |
|------------|-------|-------|---------|---------------|-------------|-----------|----------|
| 1          | 0.1   | 0.99  | 1.0     | 0.999         | 0.01        | 1,102,890 | 7,419    |
| 2          | 0.2   | 0.95  | 0.9     | 0.995         | 0.05        | 725,061   | 5,665    |
| 3          | 0.05  | 0.99  | 1.0     | 0.991         | 0.001       | 763,891   | 7,013    |

**Best Hyperparameters**: Experiment 2 achieved the best balance between the number of steps and training time.
