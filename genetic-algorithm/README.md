# Knapsack Problem Solver with Genetic Algorithm

## Overview
This project implements a solution to the Knapsack Problem using a genetic algorithm. It aims to find the optimal set of items that fit within a given capacity while maximizing the total value.
# Pseudocode

## Algorithm Description:
1. **Randomly initialize populations**: Generate an initial population of potential solutions.
2. **Determine fitness of population**: Evaluate the fitness of each individual in the population.
3. **Until convergence, repeat**:
   a. **Select parents**: Choose individuals from the population to serve as parents for the next generation.
   b. **Crossover**: Perform crossover operations to generate a new population.
   c. **Mutation**: Apply mutation to introduce variations in the population.
   d. **Calculate fitness**: Evaluate the fitness of the new population.

## Pseudocode:
```python
GA():
    initialize_population()
    find_fitness_of_population()
    
    while (termination_criteria_is_reached):
        parent_selection()
        crossover_with_probability(pc)
        mutation_with_probability(pm)
        decode_and_fitness_calculation()
        survivor_selection()
        
    find_best_solution()
    return best_solution
```
