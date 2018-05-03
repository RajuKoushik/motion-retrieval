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

numpyArray = array(list_action)

print (len(list_action))

numArray = numpyArray.reshape(-1, 1)

kmeans = KMeans(n_clusters=100, random_state=0).fit(numArray)

print (kmeans.labels_)
label = kmeans.labels_
label_list = label.tolist()
print(len(label_list))

# here's an interesting information
counter = collections.Counter(label_list)
print(counter)

pred = kmeans.predict(numArray)
poi = "poi"

print (kmeans.cluster_centers_)

total_sum = 0

# walk model

sum_walk = 0

for p in range(0, len(random_list)):
    if (random_list[p] >= 40):
        break

    filename = file_prefix + str(random_list[p]) + '.txt'
    print (filename)

    lines = open(filename).read().splitlines()
    print (len(lines))
    sub_length_walk = len(lines)
    sum_walk += sub_length_walk

counter_walk = sum_walk / 20

walk_model = [[0] * 200 for i in range(200)]

for l in range(1, int(counter_walk)):
    walk_model[kmeans.labels_[l]][kmeans.labels_[l - 1]] += 1 / counter_walk

print (walk_model)

total_sum = total_sum + int(counter_walk)

# grab model

sum_grab = 0

for p in range(0, len(random_list)):
    if (random_list[p] >= 40 and random_list[p] < 80):
        filename = file_prefix + str(random_list[p]) + '.txt'
        print (filename)

        lines = open(filename).read().splitlines()
        print (len(lines))
        sub_length_grab = len(lines)
        sum_grab += sub_length_grab
    if (random_list[p] >= 80):
        break

counter_grab = sum_grab / 20

grab_model = [[0] * 200 for i in range(200)]

for l in range(total_sum, total_sum + int(counter_grab)):
    grab_model[kmeans.labels_[l]][kmeans.labels_[l - 1]] += 1 / counter_grab

print (grab_model)

total_sum = total_sum + int(counter_grab)

# watch clock model

sum_watch = 0

for p in range(0, len(random_list)):
    if (random_list[p] >= 80 and random_list[p] < 120):
        filename = file_prefix + str(random_list[p]) + '.txt'
        print (filename)

        lines = open(filename).read().splitlines()
        print (len(lines))
        sub_length_watch = len(lines)
        sum_watch += sub_length_watch
    if (random_list[p] >= 120):
        break

counter_watch = sum_watch / 20

watch_model = [[0] * 200 for i in range(200)]

for l in range(total_sum, total_sum + int(counter_watch)):
    watch_model[kmeans.labels_[l]][kmeans.labels_[l - 1]] += 1 / counter_watch

print (watch_model)

total_sum = total_sum + int(counter_watch)

# head  model

sum_head = 0

for p in range(0, len(random_list)):
    if (random_list[p] >= 120 and random_list[p] < 160):
        filename = file_prefix + str(random_list[p]) + '.txt'
        print (filename)

        lines = open(filename).read().splitlines()
        print (len(lines))
        sub_length_head = len(lines)
        sum_head += sub_length_head
    if (random_list[p] >= 160):
        break

counter_head = sum_head / 20

head_model = [[0] * 200 for i in range(200)]

for l in range(total_sum, total_sum + int(counter_head)):
    head_model[kmeans.labels_[l]][kmeans.labels_[l - 1]] += 1 / counter_head

print (head_model)

total_sum = total_sum + int(counter_head)

# phone

sum_phone = 0

for p in range(0, len(random_list)):
    if (random_list[p] >= 160 and random_list[p] < 200):
        filename = file_prefix + str(random_list[p]) + '.txt'
        print (filename)

        lines = open(filename).read().splitlines()
        print (len(lines))
        sub_length_phone = len(lines)
        sum_phone += sub_length_phone
    if (random_list[p] >= 200):
        break

counter_phone = sum_phone / 20

phone_model = [[0] * 200 for i in range(200)]

for l in range(total_sum, total_sum + int(counter_phone)):
    phone_model[kmeans.labels_[l]][kmeans.labels_[l - 1]] += 1 / counter_phone

print (phone_model)

total_sum = total_sum + int(counter_phone)

# cross arms

sum_arms = 0

for p in range(0, len(random_list)):
    if (random_list[p] >= 200 and random_list[p] < 240):
        filename = file_prefix + str(random_list[p]) + '.txt'
        print (filename)

        lines = open(filename).read().splitlines()
        print (len(lines))
        sub_length_arms = len(lines)
        sum_arms += sub_length_arms
    if (random_list[p] >= 240):
        break

counter_arms = sum_arms / 20

arms_model = [[0] * 200 for i in range(200)]

for l in range(total_sum, total_sum + int(counter_arms)):
    arms_model[kmeans.labels_[l]][kmeans.labels_[l - 1]] += 1 / counter_arms

print (arms_model)

total_sum = total_sum + int(counter_arms)

# cross seat

sum_seat = 0

for p in range(0, len(random_list)):
    if (random_list[p] >= 240 and random_list[p] < 280):
        filename = file_prefix + str(random_list[p]) + '.txt'
        print (filename)

        lines = open(filename).read().splitlines()
        print (len(lines))
        sub_length_seat = len(lines)
        sum_seat += sub_length_seat
    if (random_list[p] >= 280):
        break

counter_seat = sum_seat / 20

seat_model = [[0] * 200 for i in range(200)]

for l in range(total_sum, total_sum + int(counter_seat)):
    seat_model[kmeans.labels_[l]][kmeans.labels_[l - 1]] += 1 / counter_seat

print (seat_model)

total_sum = total_sum + int(counter_seat)

# punch

sum_punch = 0

for p in range(0, len(random_list)):
    if (random_list[p] >= 280 and random_list[p] < 320):
        filename = file_prefix + str(random_list[p]) + '.txt'
        print (filename)

        lines = open(filename).read().splitlines()
        print (len(lines))
        sub_length_punch = len(lines)
        sum_punch += sub_length_punch
    if (random_list[p] >= 320):
        break

counter_punch = sum_punch / 20

punch_model = [[0] * 200 for i in range(200)]

for l in range(total_sum, total_sum + int(counter_punch)):
    punch_model[kmeans.labels_[l]][kmeans.labels_[l - 1]] += 1 / counter_punch

print (punch_model)

total_sum = total_sum + int(counter_punch)

# kick

sum_kick = 0

for p in range(0, len(random_list)):
    if (random_list[p] >= 320 and random_list[p] < 360):
        filename = file_prefix + str(random_list[p]) + '.txt'
        print (filename)

        lines = open(filename).read().splitlines()
        print (len(lines))
        sub_length_kick = len(lines)
        sum_kick += sub_length_kick
    if (random_list[p] >= 360):
        break

counter_kick = sum_kick / 20

kick_model = [[0] * 200 for i in range(200)]

for l in range(total_sum, total_sum + int(counter_kick)):
    kick_model[kmeans.labels_[l]][kmeans.labels_[l - 1]] += 1 / counter_kick

print (kick_model)

total_sum = total_sum + int(counter_kick)

# wave

sum_wave = 0

for p in range(0, len(random_list) - 5):
    if (random_list[p] >= 360 and random_list[p] < 399):
        filename = file_prefix + str(random_list[p]) + '.txt'
        print (filename)

        lines = open(filename).read().splitlines()
        print (len(lines))
        sub_length_wave = len(lines)
        sum_wave += sub_length_wave

counter_wave = sum_wave / 20

wave_model = [[0] * 200 for i in range(200)]

for l in range(total_sum, total_sum + int(counter_wave)):
    wave_model[kmeans.labels_[l]][kmeans.labels_[l - 1]] += 1 / counter_wave

print (wave_model)

total_sum = total_sum + int(counter_wave)

# testing
# lets take a motion sequence for testing one from each of the 10 actions

# we try to predict the cluster of the skeleton using the predict function of kmeans in sckit learn library

testing_list_1 = []
testing_list_2 = []
testing_list_3 = []
testing_list_4 = []
testing_list_5 = []
testing_list_6 = []
testing_list_7 = []
testing_list_8 = []
testing_list_9 = []
testing_list_10 = []

for i in range(1, 401):
    if i in random_list:
        continue
    else:
        # this is our left over testing data which we are gonna work on
        if i < 40:
            testing_list_1.append(i)
        elif 40 <= i < 80:
            testing_list_2.append(i)
        elif 80 <= i < 120:
            testing_list_3.append(i)
        elif 120 <= i < 160:
            testing_list_4.append(i)
        elif 160 <= i < 200:
            testing_list_5.append(i)
        elif 200 <= i < 240:
            testing_list_6.append(i)
        elif 240 <= i < 280:
            testing_list_7.append(i)
        elif 280 <= i < 320:
            testing_list_8.append(i)
        elif 320 <= i < 360:
            testing_list_9.append(i)
        else:
            testing_list_10.append(i)

print(kmeans.predict(1.22))

# walk testing data

# we gotta use the first reference skeleton data which is stored in the variables SpineX, SpineY, SpineZ, linesOne

probabilities_max = []

for i in range(len(testing_list_1)):

    answer_testing_1 = []

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
        answer_testing_1.append((kmeans.predict(tempSum))[0])

    test_probability_1 = 1
    test_probability_2 = 1
    test_probability_3 = 1
    test_probability_4 = 1
    test_probability_5 = 1
    test_probability_6 = 1
    test_probability_7 = 1
    test_probability_8 = 1
    test_probability_9 = 1
    test_probability_10 = 1

    for l in range(1, len(answer_testing_1)):

        if ((walk_model[answer_testing_1[l]][answer_testing_1[l - 1]]) * 1000000 != 0.000):
            test_probability_1 = test_probability_1 + math.log(
                walk_model[answer_testing_1[l]][answer_testing_1[l - 1]])
            print(test_probability_1)

        else:
            test_probability_1 = test_probability_1 + math.log((0.0001))

        if ((grab_model[answer_testing_1[l]][answer_testing_1[l - 1]]) * 10000000 != 0.000):
            test_probability_2 = test_probability_2 + math.log(
                grab_model[answer_testing_1[l]][answer_testing_1[l - 1]])
        else:
            test_probability_2 = test_probability_2 + math.log((0.0001))

        if ((watch_model[answer_testing_1[l]][answer_testing_1[l - 1]]) * 100000 != 0.000):
            test_probability_3 = test_probability_3 + math.log(
                watch_model[answer_testing_1[l]][answer_testing_1[l - 1]])
        else:
            test_probability_3 = test_probability_3 + math.log((0.0001))

        if (head_model[answer_testing_1[l]][answer_testing_1[l - 1]] != 0.000):
            test_probability_4 = test_probability_4 + math.log(
                head_model[answer_testing_1[l]][answer_testing_1[l - 1]])
        else:
            test_probability_4 = test_probability_4 + math.log((0.0001))

        if (phone_model[answer_testing_1[l]][answer_testing_1[l - 1]] != 0.000):
            test_probability_5 = test_probability_5 + math.log(
                phone_model[answer_testing_1[l]][answer_testing_1[l - 1]])
        else:
            test_probability_5 = test_probability_5 + math.log((0.0001))

        if (arms_model[answer_testing_1[l]][answer_testing_1[l - 1]] != 0.000):
            test_probability_6 = test_probability_6 + math.log(
                arms_model[answer_testing_1[l]][answer_testing_1[l - 1]])
        else:
            test_probability_6 = test_probability_6 + math.log((0.0001))

        if (seat_model[answer_testing_1[l]][answer_testing_1[l - 1]] != 0.000):
            test_probability_7 = test_probability_7 + math.log(
                seat_model[answer_testing_1[l]][answer_testing_1[l - 1]])
        else:
            test_probability_7 = test_probability_7 + math.log((0.0001))

        if (punch_model[answer_testing_1[l]][answer_testing_1[l - 1]] != 0.000):
            test_probability_8 = test_probability_8 + math.log(
                punch_model[answer_testing_1[l]][answer_testing_1[l - 1]])
        else:
            test_probability_8 = test_probability_8 + math.log((0.0001))

        if (kick_model[answer_testing_1[l]][answer_testing_1[l - 1]] != 0.0000):
            test_probability_9 = test_probability_9 + math.log(
                kick_model[answer_testing_1[l]][answer_testing_1[l - 1]])
        else:
            test_probability_9 = test_probability_9 + math.log((0.0001))

        if (wave_model[answer_testing_1[l]][answer_testing_1[l - 1]] != 0.00000):
            test_probability_10 = test_probability_10 + math.log(
                wave_model[answer_testing_1[l]][answer_testing_1[l - 1]])
        else:
            test_probability_10 = test_probability_10 + math.log((0.0001))

    # print('Printing the probablilities')
    # print(test_probability_1)
    # print(test_probability_2)
    # print(test_probability_3)
    # print(test_probability_4)
    # print(test_probability_5)
    # print(test_probability_6)
    # print(test_probability_7)
    # print(test_probability_8)
    # print(test_probability_9)
    # print(test_probability_10)

    probabilities_max.append(
        [test_probability_1, test_probability_2, test_probability_3, test_probability_4, test_probability_5,
         test_probability_6, test_probability_7, test_probability_8, test_probability_9, test_probability_10])

# print (linesOne)

# print (answer_testing_1)

# calculating accuracy

counter_probability = 0

for i in range(0, len(probabilities_max)):
    if max(probabilities_max[i]) == probabilities_max[i][0]:
        counter_probability += 1
    print(max(probabilities_max[i]))
    print(probabilities_max[i][0])

print('accuracy')
print(counter_probability)
print('-----')
print(len(probabilities_max))
print('==')
print(counter_probability / len(probabilities_max))

print('accuracy of 2nd model')
