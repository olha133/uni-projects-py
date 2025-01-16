import random
import matplotlib.pyplot as plt


class Room:

    def __init__(self, width, length):
        self.width = width
        self.length = length
        self.room = list()
        for _ in range(self.length):
            self.room.append(list())
            for _ in range(self.width):
                self.room[-1].append(0)

    def add_dust(self, count):

        for _ in range(0, count):
            y = random.randint(0, self.length - 1)
            x = random.randint(0, self.width - 1)
            self.room[y][x] += 1

    def has_position(self, pos):

        return 0 <= pos[0] < self.width and 0 <= pos[1] < self.length


    def has_dust(self, pos):

        if self.has_position(pos) is False:
            raise ValueError("This position does not exist in Room.has_dust.")
        return self.room[pos[1]][pos[0]] > 0


    def pickup_dust(self, pos):

        if self.has_position(pos) is True:
            if self.has_dust(pos) is True:
                self.room[pos[1]][pos[0]] -= 1

    def is_clean(self):

        for y in range(0, self.length):
            for x in range(0, self.width):
                if self.room[y][x] > 0:
                    return False
        return True


class VacuumCleaner:

    def __init__(self, start_pos, room):
        self.current_position = start_pos
        self.possible_directions = ['N', 'E', 'S', 'W']
        self.room = room

    def move(self, direction):

        if self.room.has_dust(self.current_position):
            self.room.pickup_dust(self.current_position)
            return
        if direction not in self.possible_directions:
            raise ValueError("This direction doesn't exist in VacuumCleaner.move.")
        pos = list(self.current_position)
        if direction == "E":
            pos[0] -= 1
            if self.room.has_position(tuple(pos)):
                self.current_position = tuple(pos)

        if direction == "W":
            pos[0] += 1
            if self.room.has_position(tuple(pos)):
                self.current_position = tuple(pos)

        if direction == "N":
            pos[1] -= 1
            if self.room.has_position(tuple(pos)):
                self.current_position = tuple(pos)

        if direction == "S":
            pos[1] += 1
            if self.room.has_position(tuple(pos)):
                self.current_position = tuple(pos)


def simulate_cleaning(room_dimensions, dust_count, simulation_no):
    all_steps = list()
    for _ in range(0, simulation_no):
        count_steps = 0
        room = Room(room_dimensions[0], room_dimensions[1])
        room.add_dust(dust_count)
        y = random.randint(0, room_dimensions[1] - 1)
        x = random.randint(0, room_dimensions[0] - 1)
        robot = VacuumCleaner((x, y), room)
        while room.is_clean() is False:
            robot.move(random.choice(robot.possible_directions))
            count_steps +=1
        all_steps.append(count_steps)

    return all_steps


def main():

    plt.title("Cleaning the room with a robot vacuum cleaner")
    plt.xlabel("Steps")
    plt.ylabel("Number of simulations")
    plt.style.use("dark_background")
    simulation = simulate_cleaning((5, 3), 50, 50)
    plt.hist(simulation, edgecolor = "black")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
