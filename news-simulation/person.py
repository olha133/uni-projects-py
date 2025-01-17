import random


class Person:
    """Represent one person from our group of potential readers."""

    def __init__(self, threshold, interested_in):
        self.threshold = threshold
        self.interested_in = interested_in
        self.friends_list = list()
        self.has_read = list()

    def is_interested_in(self, category):
        """Return True if person is interested in the news."""
        return category in self.interested_in

    def has_read_news(self, news):
        """Return True if person has already read the news."""
        return news in self.has_read

    def make_friends(self, population, n):
        """Populate a given person's friends list."""
        self.friends_list = random.sample(population, n)

    def process_news(self, news, time_step):
        """
        Simulate of what a person will do if they receive news.

        - if he has read this news before, he will not read it again.
        - if the message has a category that the person is not interested in,
        they will not read this news.
        - if this news does not interest him by its current level of surprise,
        he will not read it.
        - Otherwise, they will read the message and forward it
        to all their friends who are interested in the subject of this news.
        """
        if self.has_read_news(news):
            return []
        if self.is_interested_in(news.category) is False:
            return []
        if self.threshold > news.get_excitement(time_step):
            return []
        self.has_read.append(news)
        result = list()
        for friend in self.friends_list:
            if friend.is_interested_in(news.category):
                result.append(friend)
        return result
