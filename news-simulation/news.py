CATEGORIES = ["politics", "world", "culture", "tech", "local", "sport"]


class News:
    """Define the representation of the news in our simulation."""

    def __init__(self, category, excitement_rate, validity_length, created):
        self.check_data(category, excitement_rate, validity_length, created)

        self.category = category
        self.excitement_rate = excitement_rate
        self.validity_length = validity_length
        self.created = created

    def check_data(self, category, excitement_rate, validity_length, created):
        """Check the validity of parameter values."""
        if not isinstance(category, str):
            raise TypeError("Category must be a string in News.__init__.")
        if category not in CATEGORIES:
            raise ValueError("{} is not a possible category type in News.__init__.".format(category))

        if not isinstance(excitement_rate, float):
            raise TypeError("Excitement rate must be a float in News.__init__.")

        if (0 <= excitement_rate <= 1) is False:
            raise ValueError("Excitement rate must be between 0 and 1 in News.__init__.")

        if not isinstance(validity_length, int):
            raise TypeError("Validity length must be an int in News.__init__.")
        if (1 <= validity_length <= 10) is False:
            raise ValueError("Validity length must be between 1 and 10 in News.__init__.")

        if not isinstance(created, int):
            raise TypeError("Created must be an int in News.__init__.")
        if created < 1:
            raise ValueError("Created cannot be less than 1 in News.__init__.")

    def get_excitement(self, time_step):
        """Return the current excitement rate at a given point in time."""
        times = time_step - self.created
        if times > self.validity_length:
            return 0.0
        else:
            return self.excitement_rate ** times
