import numpy as np
import random
import time
import matplotlib.pyplot as plt
import pickle
from eight_puzzle_env import EightPuzzleEnv
    
class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.999, min_epsilon=0.01):
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for epsilon
        self.min_epsilon = min_epsilon  # Minimum value of epsilon
        # Initialize Q-table with dimensions for representative states and actions
        num_representative_states = 1000  # Adjust this value based on available memory and computational resources
        num_actions = env.action_space.n
        self.q_table = np.zeros((num_representative_states, num_actions))

    def hash_state(self, state):
        # Simple hashing function to map a state to a representative state index
        return hash(str(state)) % len(self.q_table)
    
    def choose_action(self, state):
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < self.epsilon:
            # Explore: choose a random action
            return self.env.action_space.sample()
        else:
            # Exploit: choose the action with the highest Q-value for the current state
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        # Q-learning update rule
        max_next_q_value = np.max(self.q_table[next_state])
        self.q_table[state, action] += self.alpha * (reward + self.gamma * max_next_q_value - self.q_table[state, action])

    def train(self, num_episodes):
        rewards = []
        start_time = time.time()
        
        for episode in range(1, num_episodes + 1):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                # Map the current state to a representative state index
                hashed_state = self.hash_state(state)

                action = self.choose_action(hashed_state)
                next_state, reward, done, _ = self.env.step(action)
                
                # Map the next state to a representative state index
                hashed_next_state = self.hash_state(next_state)

                self.update_q_table(hashed_state, action, reward, hashed_next_state)
                state = next_state
                total_reward += reward
            
            rewards.append(total_reward)
            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            print(f"Episode {episode}/{num_episodes}, Total Reward: {total_reward}")

        training_time = time.time() - start_time
        return rewards, training_time

def test_policy(agent, num_episodes):
    successes = 0
    steps_to_solve = []

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        steps = 0

        while not done:
            action = agent.choose_action(agent.hash_state(state))
            state, _, done, _ = env.step(action)
            steps += 1

        if np.array_equal(state, env.goal_state):
            print("Done: ", successes+1)
            successes += 1
            steps_to_solve.append(steps)

    success_rate = successes / num_episodes
    avg_steps_to_solve = np.mean(steps_to_solve) if steps_to_solve else None
    return success_rate, avg_steps_to_solve

# Evaluate the trained agent on test episodes
if __name__ == "__main__":
    env = EightPuzzleEnv()
    # Load the agent from the saved file
    filename = r"agents\Q-learning_agent.pkl"
    with open(filename, 'rb') as file:
        agent = pickle.load(file)
    num_episodes_test = 10
    success_rate_test, avg_steps_to_solve_test = test_policy(agent, num_episodes_test)
    print(f"Success Rate on Test Episodes: {success_rate_test:.2f}")
    print(f"Average Steps to Solve (Test): {avg_steps_to_solve_test}" if avg_steps_to_solve_test is not None else "No successful solutions in test episodes")