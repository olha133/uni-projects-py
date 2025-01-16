### Robot Vacuum Cleaner

The task is to create a simulation of the random motion of a robot vacuum cleaner in a single room, exploring how long it takes the vacuum cleaner to collect dust. Each simulation contains one room and one vacuum cleaner. There are two classes in the script for implementing the simulation:

#### Class: `Room`

This class defines a room, determining its dimensions and representing it as a two-dimensional array.

##### Attributes:
- `width` (int): The width of the room.
- `length` (int): The length of the room.
- `room` (2D array): A two-dimensional array representing the room with dimensions `width x length`. Initially, all positions contain a value of 0.

##### Methods:

1. **`add_dust(count)`**:
    - Adds a specified number of dust particles at random positions in the room.
    - The parameter `count` is a positive integer, representing the number of dust particles to be added.
    - Multiple particles can be added to the same position. The value at each position in the `room` array is incremented by 1 each time a dust particle is added.

2. **`has_position(pos)`**:
    - Determines whether a position with the given coordinates exists in the room.
    - The parameter `pos` is a pair of x and y coordinates.
    - The function returns `True` if the position is valid, otherwise returns `False`.

3. **`has_dust(pos)`**:
    - Determines whether there is dust at the given position (value greater than 0).
    - The parameter `pos` is a pair of x and y coordinates.
    - If the position is invalid, the function raises a `ValueError`.
    - Returns `True` if there is dust at the given position, otherwise returns `False`.

4. **`pickup_dust(pos)`**:
    - Removes one dust particle from a given position if the position exists and contains dust (value greater than 0).
    - The parameter `pos` is a pair of x and y coordinates.
    - The function decrements the value of the position by 1 when a dust particle is removed.

5. **`is_clean()`**:
    - Determines whether the room is clean, i.e., there is no dust (value greater than 0) at any position.
    - Returns `True` if the room is clean, otherwise returns `False`.
    - The function checks all positions in the `room` array.
#### Class: `VacuumCleaner`

This class represents the vacuum cleaner, which is defined by three values:

##### Attributes:
- `current_position` (tuple): The current position of the vacuum cleaner in the room, represented as a pair of integer values (x, y).
- `possible_directions` (list): A list of all possible directions the vacuum cleaner can move. For example, `["up", "down", "left", "right"]`.
- `room` (object of type `Room`): A reference to the `Room` object representing the room being cleaned.

##### Method:

- **`move(direction)`**:
  - This method handles the vacuum cleaner's movement.
  - **Step 1:** If there is a dust particle at the vacuum cleaner's current position, the method calls the appropriate `Room` class method to remove one dust particle. The function will stop immediately after removing dust.
  - **Step 2:** Based on the `direction` parameter, the method calculates the new position from the current position. If the direction is invalid (not in `possible_directions`), a `ValueError` is raised.
  - **Step 3:** If the new position is within the room (valid coordinates), it updates the `current_position` of the vacuum cleaner.

#### Class: `Room`

- This class represents the room, which defines the dimensions and contains dust particles.
  - The `Room` class methods are used to check for dust, remove dust, and determine the validity of a position.

### Function: `simulate_cleaning`

This method performs multiple simulations of the vacuum cleaner in a room and records the number of steps required to clean the room.

#### Parameters:
- `room_dimensions` (tuple): A tuple containing the width and length of the room (width, length).
- `dust_count` (int): The number of dust particles to add to the room before running the simulation.
- `simulation_no` (int): The number of simulations to run.

#### Function Behavior:
1. **Simulate each cleaning session:**
   - Create a new `Room` object with the specified dimensions and add `dust_count` dust particles to the room.
   - Place the vacuum cleaner at a random position within the room.
   - While there is dust in the room, the vacuum cleaner randomly chooses a direction to move and calls the `move()` method to clean the dust.
2. **Return Value:**
   - The method returns a list of integers, where each integer represents the number of steps needed to completely clean the room in one simulation.
