# News Spread Simulation

This project simulates the dynamics of news propagation within a population, analyzing how factors like excitement rate, personal interest, and social connections influence news readership.

## Features
- News Representation: Models news with attributes like category, excitement rate, and validity length.
- Population Dynamics: Simulates a group of individuals with interests, thresholds, and social connections.
- Spread Analysis:
  - Simulates the spread of news in a population.
  - Evaluates the average spread of news based on excitement rate.
  - Determines the excitement rate needed to reach a target percentage of the population.

## Categories
The simulation supports the following news categories:
- politics, world, culture, tech, local, sport.

## Core Classes
### News
- **Attributes:**
  - `category`: Category of the news (e.g., politics, tech, sports).
  - `excitement_rate`: Initial excitement rate of the news (0.0 to 1.0).
  - `validity_length`: The number of time steps the news remains valid.
  - `created`: Time step when the news was created.
- **Methods:**
  - `get_excitement(time_step)`: Returns the current excitement rate based on time.

### Person
- **Attributes:**
  - `threshold`: Minimum excitement rate required to read news.
  - `interested_in`: Categories of interest.
  - `friends_list`: List of friends.
  - `has_read`: List of news already read.
- **Methods:**
  - `process_news(news, time_step)`: Determines if a person reads and forwards news.

### Population
- **Methods:**
  - `introduce_news(news)`: Introduces news to the population.
  - `update_news(time_step)`: Updates the list of active news.
  - `count_readers(news)`: Counts how many people read a specific news item.

### Simulation
- `simulate_spread(all_news, population)`: Simulates news spreading through the population.
- `average_spread_with_excitement_rate()`: Calculates average readers for a given excitement rate.
- `excitement_to_reach_percentage()`: Determines the excitement rate needed to reach a target readership percentage.
