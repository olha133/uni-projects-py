import random
import numpy as np
from news import News, CATEGORIES
from population import Population, HomogeneousPopulation


def simulate_spread(all_news, population):
    """Examine the dynamics of news spread in the population."""
    will_read = dict()
    first_see = list()
    first_read = dict()
    second_read = dict()
    time_step = 1

    for news in all_news:
        will_read[news] = [0]
        first_read[news] = []
        first_see = population.introduce_news(news)
        for person in first_see:
            first_read[news] += person.process_news(news, time_step)

    population.update_news(time_step)
    time_step += 1

    while len(population.active_news) > 0:
        for news in all_news:
            will_read[news].append(population.count_readers(news))
            second_read[news] = []
            for person in first_read[news]:
                second_read[news] += person.process_news(news, time_step)
            first_read[news] = second_read[news].copy()
        population.update_news(time_step)
        time_step += 1

    return will_read


def average_spread_with_excitement_rate(
        excitement_rate, pop_size, friends_count, test_count=100):
    """
    Find the average number of people who will read news.

    *With a given excitement rate during the simulation.
    """
    will_read = list()
    for _ in range(0, test_count):
        news = News(random.choice(CATEGORIES), excitement_rate, 10, 1)
        population = Population(pop_size, friends_count)
        simulate_spread([news], population)
        will_read.append(population.count_readers(news))

    return will_read, sum(will_read) / len(will_read)


def excitement_to_reach_percentage(percentage, pop_size, friends_count):
    """
    Show what excitement rate news needs to have.

    *To reach a certain percentage of readers in class Population.
    """
    excitement_rate = np.float("0.0")
    red_flag = False
    while excitement_rate < 1:
        population = Population(pop_size, friends_count)
        news = News(random.choice(CATEGORIES), excitement_rate, 10, 1)
        ppl_interested = population.get_number_of_interested(news.category)
        simulate_spread([news], population)
        ppl_read = population.count_readers(news)
        if (ppl_read / ppl_interested) >= percentage:
            red_flag = True
            break
        excitement_rate += np.float("0.01")

    if red_flag is False:
        excitement_rate = None
    return excitement_rate


def excitement_to_reach_percentage_special_interest(
        percentage, pop_size, friends_count, news_category):
    """
    Show what excitement rate news needs to have.

    *To reach a certain percentage of readers in class HomogeneousPopulation.
    """
    excitement_rate = np.float("0.0")
    red_flag = False
    while excitement_rate < 1:
        pop = HomogeneousPopulation(pop_size, friends_count, news_category)
        news = News(news_category, excitement_rate, 10, 1)
        ppl_interested = pop.get_number_of_interested(news_category)
        simulate_spread([news], pop)
        ppl_read = pop.count_readers(news)
        if (ppl_read / ppl_interested) >= percentage:
            red_flag = True
            break
        excitement_rate += np.float("0.01")

    if red_flag is False:
        excitement_rate = None
    return excitement_rate


if __name__ == '__main__':
    pass
