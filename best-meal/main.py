import pandas as pd
from itertools import product


def load_meals(file_path):
    list_of_meals = pd.read_csv(file_path)
    list_of_meals.fillna(0)
    return list_of_meals


def split_into_categories(meal_list):
    grouped = meal_list.groupby(meal_list.category)
    df_soup = grouped.get_group("soup")
    df_main_dish = grouped.get_group("main dish")
    df_side_dish = grouped.get_group("sidedish")
    df_dessert = grouped.get_group("dessert")
    return df_soup, df_main_dish, df_side_dish, df_dessert


def check_intervals(intervals):
    if type(intervals) != dict:
        raise TypeError("Intervals must be loaded as dictionary")
    key_list = ["calories", "protein", "fat", "carbs", "price"]
    for key in key_list:
        if key not in intervals.keys():
            raise KeyError("Missing expected key {}".format(key))
    nlist = list(intervals.values())
    for i in range(0, 5):
        if type(nlist[i]) != tuple:
            raise TypeError("Interval limits should be loaded as tuples")
        if len(nlist[i]) != 2:
            raise ValueError(
                "Interval limits should be loaded as tuples of two values")
        if (nlist[i][1] < nlist[i][0]):
            raise ValueError("Upper limit cannot be smaller than lower limit")
    for key in intervals:
        temp = intervals.get(key)
        if key != 'price' and (type(temp[0]) != int or type(temp[1]) != int):
            raise TypeError(
                "{} limits should be set as int".format(key.capitalize()))
        if key == 'price' and (type(temp[0]) != float or type(temp[1]) != float):
            raise TypeError(
                "{} limits should be set as float".format(key.capitalize()))
    for i in range(0, 5):
        if (nlist[i][1] < nlist[i][0]):
            raise ValueError("Upper limit cannot be smaller than lower limit")


def load_intervals(file_path):
    dict_int = {}
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            alist = line.split(sep=",")
            if (alist[0] != "price"):
                pair = int(alist[1]), int(alist[2])
                dict_int[alist[0]] = pair
            if (alist[0] == "price"):
                if(len(alist) == 2):
                    pair = float(0), float(alist[1])
                    dict_int[alist[0]] = pair
                else:
                    pair = float(alist[1]), float(alist[2])
                    dict_int[alist[0]] = pair
    check_intervals(dict_int)
    return dict_int


def calculate_stats(meal):
    calories, protein, fat, carbs, amount = meal.iloc[2], meal.iloc[3],\
        meal.iloc[4], meal.iloc[5], meal.iloc[6]
    calories, protein, fat, carbs = calories * (amount/100), protein*(amount/100), \
        fat*(amount/100), carbs*(amount/100)
    return calories, protein, fat, carbs


def isclose(a, b, rel_tol=1e-09, abs_tol=0.001):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def evaluate_lunch(soup, main, side, dessert, intervals):
    all_price = get_lunch_price(soup, main, side, dessert)
    all_calories = calculate_stats(soup)[0] + calculate_stats(main)[0] \
        + calculate_stats(side)[0] + calculate_stats(dessert)[0]
    all_protein = calculate_stats(soup)[1] + calculate_stats(main)[1] \
        + calculate_stats(side)[1] + calculate_stats(dessert)[1]
    all_fat = calculate_stats(soup)[2] + calculate_stats(main)[2] \
        + calculate_stats(side)[2] + calculate_stats(dessert)[2]
    all_carbs = calculate_stats(soup)[3] + calculate_stats(main)[3] \
        + calculate_stats(side)[3] + calculate_stats(dessert)[3]
    if (intervals.get("calories")[0] <= all_calories <= intervals.get("calories")[1]) \
        and (intervals.get("protein")[0] <= all_protein <= intervals.get("protein")[1]) \
            and (intervals.get("fat")[0] <= all_fat <= intervals.get("fat")[1]) \
                and (intervals.get("carbs")[0] <= all_carbs <= intervals.get("carbs")[1]):
        if (intervals.get("price")[0] <= all_price <= intervals.get("price")[1]) \
            or isclose(all_price, intervals.get("price")[0]) \
                or isclose(all_price, intervals.get("price")[1]):
            return True
        return False
    return False


def get_lunch_price(soup, main, side, dessert):
    return soup[7]+main[7]+side[7]+dessert[7]


def generate_combinations(soups, mains, sides, desserts):
    combination_list = list()
    list_soup = [soups.iloc[y] for y in range(len(soups))]
    list_main = [mains.iloc[y] for y in range(len(mains))]
    list_side = [sides.iloc[y] for y in range(len(sides))]
    list_dessert = [desserts.iloc[y] for y in range(len(desserts))]
    for items in product(list_soup, list_main, list_side, list_dessert):
        combination_list.append(items)
    return combination_list


def find_best_meal(soups, mains, sides, desserts, intervals):
    all_combo = generate_combinations(soups, mains, sides, desserts)
    temp_list = []
    for y in range(0, len(all_combo)):
        if (evaluate_lunch(all_combo[y][0], all_combo[y][1], all_combo[y][2], \
                           all_combo[y][3], intervals) == True):
            temp_list = [all_combo[y][0], all_combo[y][1], all_combo[y][2], \
                         all_combo[y][3]]
            break
    if not temp_list:
        return None, float('inf')
    if y == len(all_combo)-1 and len(temp_list) > 0:
        result = list(pd.Series(data=temp_list))
        result_price = get_lunch_price(
            temp_list[0], temp_list[1], temp_list[2], temp_list[3])
        return result, result_price
    else:
        y += 1
        for y in range(y, len(all_combo)):
            if (evaluate_lunch(all_combo[y][0], all_combo[y][1], \
                               all_combo[y][2], all_combo[y][3], intervals)):
                current_price = get_lunch_price(temp_list[0], temp_list[1], \
                                                temp_list[2], temp_list[3])
                next_price = get_lunch_price(all_combo[y][0], all_combo[y][1], \
                                             all_combo[y][2], all_combo[y][3])
                if (next_price < current_price) \
                    and isclose(current_price, next_price, abs_tol=0.001) == False:
                    temp_list = [all_combo[y][0], all_combo[y][1], all_combo[y][2], all_combo[y][3]]
    result = list(pd.Series(data=temp_list))
    result_price = get_lunch_price(temp_list[0], temp_list[1], temp_list[2], temp_list[3])
    return result, result_price


def main(meal_file_path, interval_file_path):
    meal_df = load_meals(meal_file_path)
    intervals = load_intervals(interval_file_path)

    soups, mains, sides, desserts = split_into_categories(meal_df)

    return find_best_meal(soups, mains, sides, desserts, intervals)


if __name__ == '__main__':
    result = main(
        '1a_sample_meals.csv',
        '1a_sample_interval.txt'
    )
    print(result)
