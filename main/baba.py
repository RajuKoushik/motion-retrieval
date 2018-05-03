import numpy as np
from numpy import array
import collections
import matplotlib.pyplot as plt
import sys
import numpy as np
from numpy import array
import collections
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import KMeans

import math

sys.path.append("../tools/")
import pylab as pl


def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


import random

# generating random numbers


random_list1 = random.sample(range(1, 400), 300)

random_list = sorted(random_list1)

print (random_list)

bigX = [] * 10000000

file_prefix = 'action_'

answer = [] * 1000000000

listOne = [] * 10000

linesOne = [0] * 10000

# lines one has the first reference skeleton

spineX = 0
spineY = 0
spineZ = 0

list_action = []

for i in range(0, len(random_list)):
    filename = file_prefix + str(random_list[i]) + '.txt'
    print (filename)

    lines = open(filename).read().splitlines()
    print (lines[0])

    flag = 0

    for j in range(int(len(lines) / 20) - 20):

        tempSum = 0.00

        for k in range(0, 19):
            line1 = lines[k + flag].split(',')
            line2 = lines[k + 20 + flag].split(',')

            tempSum = tempSum + ((num(line1[0]) - num(line2[0])) ** 2 + (num(line1[1]) - num(line2[1])) ** 2 + (
                num(line1[2]) - num(line2[2])) ** 2) ** (0.5)

        list_action.append(tempSum)

        flag = flag + 20

print(list_action)