### Nearest Neighbor Algorithm for TSP

This Python script solves the Traveling Salesman Problem (TSP) using the Nearest Neighbor algorithm.

#### Function: `nearest_neighbor(file)`
- **Description**: Solves the TSP by finding the shortest possible route based on a greedy approach, starting from a random city and visiting the nearest unvisited city at each step.
- **Parameters**: 
  - `file`: Path to a text file containing the distance data for cities.
- **Steps**:
  1. Read the input file and parse the distance matrix into a list.
  2. Select a random starting city.
  3. Iteratively find and visit the nearest city (based on the minimum travel cost).
  4. Avoid revisiting cities.
  5. Once all cities are visited, return to the starting city and compute the total cost.
  6. Output the result (total cost and route) to a file named `Result.txt`.

#### Example Usage:
```python
nearest_neighbor("tsp_examples/example3.txt")
