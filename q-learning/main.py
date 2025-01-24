import gym
import matplotlib.pyplot as plt
import numpy as np
import random

class QLearning:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, epsilon=1.0):
        """
        Initialize the QLearning agent.

        Args:
            env (gym.Env): The environment.
            learning_rate (float): The learning rate for updating Q-values.
            discount_factor (float): The discount factor for future rewards.
            epsilon (float): The exploration rate for epsilon-greedy action selection.
        """
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.num_actions = 3  # Assuming 3 actions for MountainCarContinuous
        self.max_episodes = 50000
        self.interval = 1000
        
        # Discretize the observation and action spaces
        self.pos_space = np.linspace(self.env.observation_space.low[0], self.env.observation_space.high[0], 18)
        self.vel_space = np.linspace(self.env.observation_space.low[1], self.env.observation_space.high[1], 28)
        self.act_space = np.linspace(self.env.action_space.low, self.env.action_space.high, self.num_actions)

        # Initialize Q-table
        self.q_table = self.make_q_table()

    def get_state(self, state):
        """
        Discretize the continuous state space.

        Args:
            state (tuple): The continuous state.

        Returns:
            tuple: The discretized state.
        """
        pos, vel = state
        discretized_pos = np.digitize(pos, self.pos_space)
        discretized_vel = np.digitize(vel, self.vel_space)
        return (discretized_pos, discretized_vel)

    def make_q_table(self):
        """
        Create the Q-table.

        Returns:
            dict: The Q-table.
        """
        col1, col2 = np.meshgrid(range(len(self.pos_space) + 1),
                             range(len(self.vel_space) + 1))
        states = np.column_stack((col1.flatten(), col2.flatten()))
        q_table = {}

        for state in states:
            for action in self.act_space:
                q_table[tuple(state), tuple(action)] = 0
        return q_table

    def max_action(self, state, actions):
        """
        Get the action with the maximum Q-value for a given state.

        Args:
            state (tuple): The state.
            actions (list): List of available actions.

        Returns:
            float: The action with the maximum Q-value.
        """
        q_values = [self.q_table[state, tuple(action)] for action in actions]
        max_index = np.argmax(q_values)
        return actions[max_index]

    def choose_action(self, state, q_table, epsilon):
        """
        Choose an action based on epsilon-greedy policy.

        Args:
            state (tuple): The current state.
            q_table (dict): The Q-table.
            epsilon (float): The exploration rate.

        Returns:
            int: The selected action.
        """
        if random.uniform(0, 1) < epsilon:
            action_index = random.randint(0, len(self.act_space) - 1)
            return self.act_space[action_index]
        else:
            return self.max_action(q_table, state, self.act_space)

    def train(self, max_episode_steps = 1000):
        """
        Train the Q-learning agent.

        Args:
            max_episode_steps (int): Maximum steps per episode. Default is 1000.

        Returns:
            list: List of rewards for each episode.
            dict: The trained Q-table.
        """
        env._max_episode_steps = max_episode_steps

        rewards_q = []
        current_reward = 0

        q_table = self.make_q_table()
        
        # Loop over episodes
        for ep in range(self.max_episodes):
            # Reset the environment for a new episode
            observation, info = env.reset()
            if ep % self.interval == 0:
                print('Game number:', ep, 'Epsilon: ',
                    self.epsilon, 'Reward: ', current_reward)

            current_reward = 0
            terminated, truncated = False, False
            
            # Get the initial state
            state = self.get_state(observation)

            while not (terminated or truncated):

                # Choose an action based on the current state and Q-table
                action = self.choose_action(state, q_table, epsilon)

                # Take the chosen action and observe the next state and reward
                next_observation, reward, terminated, truncated, info = env.step(action)
                current_reward += reward

                next_state = self.get_state(next_observation)

                # Determine the next action based on the next state and Q-table
                next_action = self.max_action(q_table, next_state, self.act_space)
                
                # Update the Q-value for the current state and action
                q_table[state, tuple(action)] += self.learning_rate * (reward + self.discount_factor * q_table[next_state, tuple(next_action)] - q_table[state, tuple(action)])


                state = next_state

            # Update epsilon for the next episode
            epsilon = epsilon - 2/self.max_episodes if epsilon > 0.01 else 0.01
            rewards_q.append(current_reward)
        return rewards_q, q_table
    
    def save(self, file_name='q_table.txt'):
        """
        Save the Q-table to a file.

        Args:
            file_name (str): The name of the file to save to.
        """
        with open(file_name, 'w') as file:
            file.write(str(self.q_table))

    def show_results(self, train_rewards):
        """
        Show the training rewards plot.

        Args:
            train_rewards (list): List of rewards for each episode.
        """
        fig, ax = plt.subplots()
        ax.scatter(range(self.max_episodes), train_rewards, s=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Training Rewards')
        plt.show()

# def run_experiments(env_name='MountainCarContinuous-v0', hyperparams={}):
#     # Initialize environment
#     env = gym.make(env_name)
#     env._max_episode_steps = 1000

#     # Define hyperparameters to test
#     lr_values = hyperparams.get('learning_rate', [0.1, 0.2, 0.3])
#     discount_values = hyperparams.get('discount_factor', [0.8, 0.9, 0.95])
#     epsilon_values = hyperparams.get('epsilon', [0.1, 0.2, 0.3])

#     results = {}

#     # Perform experiments with different hyperparameter settings
#     for lr in lr_values:
#         for discount_factor in discount_values:
#             for epsilon in epsilon_values:
#                 agent = QLearning(env, learning_rate=lr, discount_factor=discount_factor, epsilon=epsilon)
#                 avg_rewards = []

#                 # Perform multiple test runs
#                 for _ in range(5):  # Adjust as needed
#                     rewards, q_table = agent.train(max_episode_steps=1000)  # Adjust episodes as needed
#                     avg_rewards.append(np.mean(rewards))

#                 # Calculate average results
#                 avg_reward = np.mean(avg_rewards)

#                 # Store results
#                 results[(lr, discount_factor, epsilon)] = avg_reward

#     return results

if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')
    agent = QLearning(env)
    rewards, q_table = agent.train(max_episode_steps=1000)
    agent.save()
    agent.show_results(rewards)
    env.close()
    # hyperparams = {
    #     'learning_rate': [0.1, 0.2, 0.3],
    #     'discount_factor': [0.8, 0.9, 0.95],
    #     'epsilon': [0.1, 0.2, 0.3]
    # }

    # results = run_experiments(env_name='MountainCarContinuous-v0', hyperparams=hyperparams)

    # # Print or visualize results
    # for params, avg_reward in results.items():
    #     print("Hyperparameters:", params, "Average Reward:", avg_reward)
