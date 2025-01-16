# -*- coding: utf-8 -*-
import random


def nearest_neighbor(file):
    '''This function solves the TSP using the Nearest Neighbor algorithm'''
    with open(file) as f:
        data = []
        for line in f:
            line = line.split()  # to deal with blank
            if line:            # skip lines
                line = [int(i) for i in line]
                data.append(line)
    data_c = data.copy()
    result = list()
    money = 0
    # on what number the file ends aka how many roads will be
    end = data[-1][0]
    start = random.randint(1, end)
    start_c = start
    result.append(start)
    road = 0
    while (road < end):
        first = 0  # first iteration?
        for x in data:  # searching for the cheapest/shortest road
            # if it's first iteration we need to find first possible road
            if (first == 0) and (x[0] == start or x[1] == start):
                first = x
                cost = x[2]
                continue
            # find next road and check cost
            if x[0] == start or x[1] == start:
                new_cost = x[2]
                if new_cost < cost:
                    first = x
                    cost = new_cost
        if first[0] != start:  # if needed, we change positions for simplicity
            first[0], first[1] = first[1], first[0]
        result.append(first[1])  # add the position where we finished
        money += first[2]  # adding cost

        for x in data.copy():  # we cannot return to previous positions
            if x[0] == start or x[1] == start:
                data.remove(x)
        start = first[1]  # now our end is a new beginning
        road += 1
    for x in data_c:
        if (x[0] == start_c or x[0] == start) \
                and (x[1] == start_c or x[1] == start):
            money += x[2]
    f = open("Result.txt", "w")
    f.write(str(money) + '\n')
    for x in range(0, end + 1):
        # write each item on a new line
        if x == end:
            f.write("%s" % result[x])
            break
        f.write("%s," % result[x])
    f.close()


nearest_neighbor("tsp_examples/example3.txt")
