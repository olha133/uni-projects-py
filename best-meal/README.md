## Problem Description

### Input Data

The input consists of a CSV file listing the offered meals, formatted as shown in the example below. The first line describes the attributes included in the database:

- **`meal`**: Name of the meal.  
- **`category`**: Type of meal; in this implementation, the categories are:
  - **`soup`**: Soups.
  - **`main dish`**: Main dishes.
  - **`sidedish`**: Side dishes.
  - **`dessert`**: Desserts.
- **`calories`**: The number of calories per 100g (or per 100ml for soups).
- **`protein`**: The amount of protein in the meal per 100g (or per 100ml for soups).
- **`fat`**: The amount of fat in the meal per 100g (or per 100ml for soups).
- **`carbs`**: The amount of carbohydrates in the meal per 100g (or per 100ml for soups).
- **`amount`**: Serving size in grams or millilitres for soups.
- **`price`**: The price of a portion.

#### Example CSV File for Meals
```csv
meal,category,calories,protein,fat,carbs,amount,price
boiled potatoes,sidedish,93,2.18,0.22,21.29,280,1.10
boiled rice,sidedish,206,4.47,4.33,38.6,190,0.60
fried chicken,main dish,369,21.9,18.3,28.7,180,2.60
goulash soup,soup,63,3.82,3.1,5.36,350,0.6
banana,dessert,90,1.2,0.24,19.8,120,0.25
```
### Task Description

A diner's lunch consists of the following components:
- **1 Soup**
- **1 Main Dish**
- **1 Side Dish**
- **1 Dessert**

The task is to write a program that selects the **cheapest possible combination** of these meals. 

### Additional Constraints
To increase complexity, the selection must also satisfy specific constraints on the total nutritional values of the lunch:
- **Calories**
- **Protein**
- **Fat**
- **Carbohydrates**

These constraints will be provided in a separate file, specifying acceptable ranges for each attribute. Below is an example of such a file:
#### Example Constraints File
```csv
calories,50,1000
protein,5,300
fat,5,300
carbs,5,300
price,4.00
```
### Input Format and Constraints

The input format for the constraints file follows **CSV rules without a header**. Each line specifies the acceptable limits for a given nutritional or financial parameter. 

#### Example Explanation:
In the provided example:
1. The eater is looking for a combination of meals that satisfies:
   - **Calories**: Between 50 and 1000 kcal
   - **Protein**: Between 5 and 300 g
   - **Fat**: Between 5 and 300 g
   - **Carbohydrates**: Between 5 and 300 g
   - **Price**: A maximum of €4, or optionally, between €2.50 and €4 if both lower and upper limits are specified.

2. If no lower limit is given for the price, it is assumed to be **0**.

#### File Characteristics:
- The order of the parameters is always the same.
- The file must include values for all parameters.

---

### Selection Process

Once the menu and bounds have been successfully loaded, the program should proceed to select the optimal combination of meals. 

#### Objective:
The selected combination (soup, main dish, side dish, and dessert) must:
1. Stay within the defined **range** for each indicator (calories, protein, fat, carbs, price).
2. Minimize the **total price**, ensuring it is:
   - **Not less than the lower limit.**
   - **Not greater than the upper limit.**

---

### Calculation of Nutritional Values
The calculations must account for the **portion size** of each dish. For example:
- Boiled potatoes: 100g contains 93 calories, but a serving size is 280g.  
  **Calories in a serving**:  
  280 g × (93 kcal / 100 g) = 260.4 kcal
  
Using similar calculations:
- **Boiled Rice**:  
  190 g × (206 kcal / 100 g) = 391.4 kcal
- **Fried Schnitzel**:  
  180 g × (369 kcal / 100 g) = 664.2 kcal
- **Goulash Soup**:  
  350 ml × (63 kcal / 100 ml) = 220.5 kcal
- **Bananas**:  
  120 g × (90 kcal / 100 g) = 108 kcal

---
### Choosing Between Side Dishes

In this case, we have a choice of two side dishes: **potatoes** or **rice**. However, if we have a **calorie limit** for the whole lunch of **1300 kcal**, we see that:

- With **potatoes**, the total calories are **1253.1 kcal**, which stays within the limit.
- With **rice**, the total calories are **1384.1 kcal**, which exceeds the calorie limit, even if combined with a cheaper lunch.

On the other hand, **potatoes** contain **less protein**, so if our diner cares more about the amount of protein and has set a higher **lower limit for protein**, they should choose **rice**.

If both the lunch with **potatoes** and the lunch with **rice** meet all the constraints, we would choose the **cheaper option**, i.e., **rice**.

### Implementation Notes
The program must:
1. Accurately compute nutritional values based on portion sizes.
2. Efficiently search for the **cheapest combination** of meals that satisfies the constraints.
3. Handle edge cases, such as when no combination fits within the specified bounds.
### Task 1: Implement the `load_meals` Method

Implement the `load_meals` method, which takes one parameter: the path to a file containing the cafeteria menu in CSV format. The first line in this file is a header with attribute names, and each subsequent line describes one item.

The `load_meals` function should return a single value: a **pandas dataframe**.

### Task 2: Implement the `split_into_categories` Method

Implement the `split_into_categories` method, which splits the dataframe loaded by the `load_meals` method into four smaller dataframes, each containing different types of meals: 

- **soups**
- **main courses**
- **side dishes**
- **desserts**

The return values of this function should be the four dataframes representing each category of meals.
### Task 3: Implement the `load_intervals` and `check_intervals` Methods

#### Task 3.1: Implement the `load_intervals` Method

Implement the `load_intervals` method that loads the bounds for the calorie, protein, fat, carbohydrate, and price variables from a `.txt` file. The method's parameter is the path to the file.

Each line in the file represents the limits of some indicator with three values: 
- Indicator name 
- Lower limit 
- Upper limit 

In the case of price, the lower limit may be missing, in which case treat it as `0`. 

The function should return a single dictionary with the retrieved data. The keys of the dictionary will be the names of the indicators, and the values will be pairs of values representing the lower and upper limits. 

- The price limit values are stored as **decimal numbers (float)**.
- The other numeric values (calories, protein, fat, carbs) are stored as **integers (int)**.

The function should return the dictionary only if all the retrieved data is valid. Otherwise, an exception should be raised. 

#### Task 3.2: Implement the `check_intervals` Method

Implement the `check_intervals` method to validate the data loaded by the `load_intervals` method. 

The function receives one parameter - the dictionary loaded by the `load_intervals` function. The `check_intervals` function does not return any value but generates an exception in case of invalid data:

- **TypeError**: "Intervals must be loaded as dictionary" if the parameter does not represent a dictionary.
- **KeyError**: "Missing expected key XY" if any of the expected keys (calories, protein, fat, carbs, price) are missing (replace XY with the missing key name).
- **TypeError**: "Interval limits should be loaded as tuples" if a value under any key is not a tuple.
- **ValueError**: "Interval limits should be loaded as tuples of two values" if a tuple under a key does not have exactly two values.
- **TypeError**: "XY limits should be set as T" if the limits are not of the correct type (for price, it must be a float; for other indicators, integers are expected).
- **ValueError**: "Upper limit cannot be smaller than lower limit" if the upper limit is smaller than the lower limit for any indicator.
### Task 4: Implement the `calculate_stats` Function

Implement the `calculate_stats` function to calculate the nutritional values for a single serving of food. 

The function receives a row from the dataframe (a Series with the retrieved data) as a parameter. For each food item, the nutritional values (calories, protein, fat, carbohydrates) are provided for 100g of the given product. You need to recalculate these values based on the portion size provided by the `amount` attribute.

The function should return four values:
- The amount of **calories**
- The amount of **protein**
- The amount of **fat**
- The amount of **carbohydrates** in a portion of the meal.

#### Example:
If the row indicates that the food contains 93 kcal per 100g and the portion size is 280g, the function should return the recalculated values for calories, protein, fat, and carbohydrates based on the portion size.

---

### Problem 5: Implement the `evaluate_lunch` Function

Implement the `evaluate_lunch` function to determine whether a combination of meals (lunch) satisfies the constraints provided by the intervals of each attribute (calories, protein, fat, carbohydrates, and price). 

The function should return `True` if the lunch is acceptable and meets all the defined constraints, or `False` if any of the limits are not met.

#### Parameters:
- `soup`: The selected soup from the menu, passed as a row from a dataframe (Series).
- `main`: The selected main dish from the menu, passed as a row from a dataframe (Series).
- `side`: The selected side dish from the menu, passed as a row from a dataframe (Series).
- `dessert`: The selected dessert from the menu, passed as a row from a dataframe (Series).
- `intervals`: A dictionary representing the limits, formatted according to the data loaded by the `load_intervals` function.

#### Example:
The function should check if the total values of calories, protein, fat, carbs, and price for the selected combination (soup, main, side, dessert) are within the specified ranges provided by the `intervals` dictionary. If any value falls outside the allowed range, the function should return `False`.
### Problem 6: Implement the `get_lunch_price` Function

Implement the `get_lunch_price` function to calculate the total price of a lunch that includes a soup, main dish, side dish, and dessert.

The selected items will be given to the function as parameters (just like in `evaluate_lunch`). The function should return the total price of the lunch as a decimal (float).

#### Parameters:
- `soup`: The selected soup from the menu (Series).
- `main`: The selected main dish from the menu (Series).
- `side`: The selected side dish from the menu (Series).
- `dessert`: The selected dessert from the menu (Series).

#### Example:
The function should sum up the prices of the selected soup, main dish, side dish, and dessert and return the total price of the lunch.

---

### Problem 7: Implement the `generate_combinations` Function

Implement the `generate_combinations` function to return a list of all possible lunches that can be combined from the menu.

The function receives four dataframes as parameters:
- A dataframe containing all possible soups.
- A dataframe containing all possible main courses.
- A dataframe containing all possible side dishes.
- A dataframe containing all possible desserts.

Each combination must have one soup, one main dish, one side dish, and one dessert. The function should represent each combination as a tuple of pandas Series (i.e., rows representing the selected menu items).

#### Example:
The function will return a list of tuples, where each tuple represents a possible lunch combination with one item from each menu category (soup, main, side, dessert).

---

### Problem 8: Implement the `find_best_meal` Function

Implement the `find_best_meal` function to find the cheapest possible lunch that meets the nutrient requirements set by the caterer.

#### Parameters:
- `soups`: List of available soups (pandas DataFrame).
- `mains`: List of available main meals (pandas DataFrame).
- `sides`: List of available side dishes (pandas DataFrame).
- `desserts`: List of available desserts (pandas DataFrame).
- `intervals`: Dictionary representing the nutrient limits (format according to the data loaded by the `load_intervals` function).

#### Returns:
The function returns two values:
- The cheapest possible lunch meeting the nutrient conditions: A list of four pandas Series (representing the selected soup, main course, side dish, and dessert). If no lunch meets the conditions, the value is `None`.
- The total price of the lunch: A decimal (float). If no lunch meets the conditions, the value is infinity.

#### Example:
The function will evaluate all possible combinations of soups, main courses, side dishes, and desserts. For each combination, it will check if the nutrient values fall within the specified intervals. It will then return the combination with the lowest price that satisfies all the conditions.
