import random
from person import Person
from news import CATEGORIES


class Population:
    """Represent a group of people who read and send news to each other."""

    def __init__(self, n, friends_count):
        self.people = list()
        self.active_news = list()

        self.generate_population(n, friends_count)

    def generate_population(self, n, friends_count):
        """Generate the population and create friendships between people."""
        for _ in range(0, n):
            self.people.append(Person(random.random(), random.sample(CATEGORIES, 4)))

        for prs in self.people:
            temp_list = self.people.copy()
            temp_list.remove(prs)
            prs.make_friends(temp_list, friends_count)

    def introduce_news(self, news):
        """Represent the moment when news starts to spread in the population."""
        result = list()
        for person in self.people:
            if len(result) == 5:
                break
            if person.is_interested_in(news.category):
                result.append(person)
                if news not in self.active_news:
                    self.active_news.append(news)

        return result

    def update_news(self, time_step):
        """Update the list of active messages."""
        for news in self.active_news:
            if news.get_excitement(time_step) == 0.0:
                self.active_news.remove(news)

    def count_readers(self, news):
        """Return the number of people that read this news."""
        count = 0
        for person in self.people:
            if person.has_read_news(news):
                count += 1
        return count

    def get_number_of_interested(self, category):
        """Return the number of people who are interested this category."""
        count = 0
        for person in self.people:
            if person.is_interested_in(category):
                count += 1
        return count


class HomogeneousPopulation(Population):
    """Represents the population when people is interested in a certain category of news."""

    def __init__(self, n, friends_count, category):
        self.category = category
        super().__init__(n, friends_count)

    def generate_population(self, n, friends_count):
        """Generate the population and create friendships between people."""
        for _ in range(0, n):
            self.people.append(Person(random.random(), self.category))
        for prs in self.people:
            temp_list = self.people.copy()
            temp_list.remove(prs)
            prs.make_friends(temp_list, friends_count)
