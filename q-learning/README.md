# Q-Learning Agent for Continuous MountainCar

This project implements a Q-Learning algorithm for the `MountainCarContinuous-v0` environment using a discretized state and action space. The agent is trained to learn an optimal policy through trial-and-error interactions with the environment.

## Features
- Discretization of continuous state and action spaces for compatibility with Q-Learning.
- Epsilon-greedy action selection for balancing exploration and exploitation.
- Adjustable hyperparameters for learning rate, discount factor, and exploration rate.
- Visualization of training rewards for monitoring performance.
## Outputs

- A plot showing the training rewards over episodes.
- The trained Q-table saved to `q_table.txt`.

## Hyperparameters

You can modify the following hyperparameters in the `QLearning` class:

- **`learning_rate`**: Controls the rate of Q-value updates (default: `0.1`).
- **`discount_factor`**: Determines the importance of future rewards (default: `0.9`).
- **`epsilon`**: Exploration rate for epsilon-greedy policy (default: `1.0`).

## Example

```python
env = gym.make('MountainCarContinuous-v0')
agent = QLearning(env, learning_rate=0.1, discount_factor=0.9, epsilon=1.0)
rewards, q_table = agent.train(max_episode_steps=1000)
agent.save()
agent.show_results(rewards)
env.close()
```
## Training Rewards

After training, a scatter plot of rewards per episode will be displayed, showing the agent's progress over time.

## Q-Table Save

The trained Q-table is saved as `q_table.txt` for later use or inspection.
