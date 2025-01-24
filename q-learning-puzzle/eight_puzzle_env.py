import gymnasium
from gymnasium import spaces
import numpy as np
from random import shuffle

class EightPuzzleEnv(gymnasium.Env):
    def __init__(self):
        self.observation_space = spaces.Box(low=0, high=8, shape=(3, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(4)  # 0: up, 1: down, 2: left, 3: right
        self._action_to_direction = {
            0: np.array([0, 1]),
            1: np.array([0, -1]),
            2: np.array([-1, 0]),
            3: np.array([1, 0]),
        }
        self.state = None #the state of the entire puzzle 
        self.goal_state = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
        self.prev_correct_numbers = set()
        self.reset()

    def reset(self, seed = None):
        super().reset(seed=seed)
        self.state = self.generate_valid_initial_state()
        return self.state

    def step(self, action):
        empty_row, empty_col = np.where(self.state == 0)
        empty_row, empty_col = empty_row[0], empty_col[0]
        
        if action == 0:  # up
            new_row = empty_row - 1
            new_col = empty_col
        elif action == 1:  # down
            new_row = empty_row + 1
            new_col = empty_col
        elif action == 2:  # left
            new_row = empty_row
            new_col = empty_col - 1
        elif action == 3:  # right
            new_row = empty_row
            new_col = empty_col + 1

        if 0 <= new_row < 3 and 0 <= new_col < 3:
            # Perform the move
            self.state[empty_row, empty_col] = self.state[new_row, new_col]
            self.state[new_row, new_col] = 0

        done = np.array_equal(self.state, self.goal_state)
        reward = self.calculate_reward(done)
        return self.state, reward, done, {}
    
    # Rewarding Only New Numbers in Correct Position:
    def calculate_reward(self, done):
        reward = -1  # Default reward for each step
        if done:  # If goal state is reached
            reward += 20  # Bonus reward for reaching goal state
        else:
            # Iterate over each number in the puzzle
            for i in range(1, 9):
                goal_row, goal_col = np.where(self.goal_state == i)
                goal_row, goal_col = goal_row[0], goal_col[0]
                
                # Check if the number is in the correct position
                if self.state[goal_row, goal_col] == i:
                    # Check if the number was previously in the correct position
                    if i not in self.prev_correct_numbers:
                        # If it's a new number in the correct position, increase the reward
                        reward += 1
                        # Add the number to the set of previously correct numbers
                        self.prev_correct_numbers.add(i)
        return reward
        
    def generate_valid_initial_state(self):
        state = np.array(range(9))
        shuffle(state)
        while not self.is_solvable(state):
            shuffle(state)
        state = state.reshape(3, 3)
        return state

    def is_solvable(self, state):
        inversion_count = 0
        # Iterate over each pair of elements in the state array
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                # Count inversions if the current pair violates the natural order and is not the empty space (0)
                if state[i] and state[j] and state[i] > state[j]:
                    inversion_count += 1
        # The puzzle is solvable if the inversion count is even
        if inversion_count % 2 == 0:
            return True
        else:
            return False

    def render(self):
        print(self.state)
