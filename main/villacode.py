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

for i in range(0, len(random_list)):
    filename = file_prefix + str(random_list[i]) + '.txt'
    print (filename)

    lines = open(filename).read().splitlines()
    print (lines[0])

    if i == 0:
        def_spine_line = lines[10]
        def_spine_first = def_spine_line.split(',')

        spineX = num(def_spine_first[0])
        spineY = num(def_spine_first[1])
        spineZ = num(def_spine_first[2])
        print ('enter the dragon')
        for ii in range(0, 20):
            linesOne[ii] = lines[ii]

    flag = 0
    print (len(lines))
    for j in range(10, (len(lines))):
        if i == 0:

            spineLineNumber = 30 + flag
            print (spineLineNumber)
        else:
            spineLineNumber = 10 + flag
            print (spineLineNumber)

        if spineLineNumber > (len(lines)):
            break

        spineLine = lines[spineLineNumber]
        spineLineArray = spineLine.split(',')

        spineTempX = num(spineLineArray[0])
        spineTempY = num(spineLineArray[1])
        spineTempZ = num(spineLineArray[2])

        diffX = spineTempX - spineX
        diffY = spineTempY - spineY
        diffZ = spineTempZ - spineZ

        coordinateX = [] * 1000
        coordinateY = [] * 1000
        coordinateZ = [] * 1000

        tempSum = 0

        for k in range(0, 20):
            listOne = linesOne[k].split(',')

            if i == 0:

                listTwo = lines[k + 20 + flag].split(',')
                print(str(k + 20 + flag) + "knkjkjk")
            else:
                listTwo = lines[k + 0 + flag].split(',')
                print(str(k + 0 + flag) + "bento")

            tempSum += (((num(listTwo[0]) - num(diffX)) - num(listOne[0])) ** (2) + (
                (num(listTwo[1]) - num(diffY)) - num(listOne[1])) ** (2) + (
                            (num(listTwo[2]) - num(diffZ)) - num(listOne[2])) ** (2)) ** (0.5)

        # print tempSum

        j += 20
        flag += 20

        answer.append(tempSum)
# print (answer)
print (linesOne)

numpyArray = array(answer)

numArray = numpyArray.reshape(-1, 1)

kmeans = KMeans(n_clusters=69, random_state=0).fit(numArray)

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

counter_walk = (sum_walk / 20) - 1

walk_model = [[0] * 200 for i in range(200)]

for l in range(1, int(counter_walk) + 1):
    walk_model[kmeans.labels_[l]][kmeans.labels_[l - 1]] += (1 / counter_walk)

print (walk_model)

total_sum = total_sum + int(counter_walk)

# grab model

sum_grab = 0

for p in range(0, len(random_list)):
    if (random_list[p] > 40 and random_list[p] <= 80):
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

for l in range(total_sum, total_sum + int(counter_grab) + 1):
    grab_model[kmeans.labels_[l]][kmeans.labels_[l - 1]] += 1 / counter_grab

print (grab_model)

total_sum = total_sum + int(counter_grab)

# watch clock model

sum_watch = 0

for p in range(0, len(random_list)):
    if (random_list[p] > 80 and random_list[p] <= 120):
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

for l in range(total_sum, total_sum + int(counter_watch) + 1):
    watch_model[kmeans.labels_[l]][kmeans.labels_[l - 1]] += 1 / counter_watch

print (watch_model)

total_sum = total_sum + int(counter_watch)

# head  model

sum_head = 0

for p in range(0, len(random_list)):
    if (random_list[p] > 120 and random_list[p] <= 160):
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

for l in range(total_sum, total_sum + int(counter_head) + 1):
    head_model[kmeans.labels_[l]][kmeans.labels_[l - 1]] += 1 / counter_head

print (head_model)

total_sum = total_sum + int(counter_head)

# phone

sum_phone = 0

for p in range(0, len(random_list)):
    if (random_list[p] > 160 and random_list[p] <= 200):
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

for l in range(total_sum, total_sum + int(counter_phone) + 1):
    phone_model[kmeans.labels_[l]][kmeans.labels_[l - 1]] += 1 / counter_phone

print (phone_model)

total_sum = total_sum + int(counter_phone)

# cross arms

sum_arms = 0

for p in range(0, len(random_list)):
    if (random_list[p] > 200 and random_list[p] <= 240):
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

for l in range(total_sum, total_sum + int(counter_arms)+1):
    arms_model[kmeans.labels_[l]][kmeans.labels_[l - 1]] += 1 / counter_arms

print (arms_model)

total_sum = total_sum + int(counter_arms)

# cross seat

sum_seat = 0

for p in range(0, len(random_list)):
    if (random_list[p] > 240 and random_list[p] <= 280):
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

for l in range(total_sum, total_sum + int(counter_seat) + 1):
    seat_model[kmeans.labels_[l]][kmeans.labels_[l - 1]] += 1 / counter_seat

print (seat_model)

total_sum = total_sum + int(counter_seat)

# punch

sum_punch = 0

for p in range(0, len(random_list)):
    if (random_list[p] > 280 and random_list[p] <= 320):
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

for l in range(total_sum, total_sum + int(counter_punch) + 1):
    punch_model[kmeans.labels_[l]][kmeans.labels_[l - 1]] += 1 / counter_punch

print (punch_model)

total_sum = total_sum + int(counter_punch)

# kick

sum_kick = 0

for p in range(0, len(random_list)):
    if (random_list[p] > 320 and random_list[p] <= 360):
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
    if (random_list[p] > 360 and random_list[p] <= 400):
        filename = file_prefix + str(random_list[p]) + '.txt'
        print (filename)

        lines = open(filename).read().splitlines()
        print (len(lines))
        sub_length_wave = len(lines)
        sum_wave += sub_length_wave

counter_wave = sum_wave / 20

wave_model = [[0] * 200 for i in range(200)]

for l in range(total_sum, total_sum + int(counter_wave) + 1):
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

    filename = file_prefix + str(testing_list_1[i]) + '.txt'
    print (filename)

    lines = open(filename).read().splitlines()
    print (lines[0])

    flag = 0
    print (len(lines))
    for j in range(10, (len(lines) - 29)):
        spineLineNumber = 30 + flag
        if spineLineNumber > len(lines):
            break

        spineLine = lines[spineLineNumber]
        spineLineArray = spineLine.split(',')

        spineTempX = num(spineLineArray[0])
        spineTempY = num(spineLineArray[1])
        spineTempZ = num(spineLineArray[2])

        diffX = spineTempX - spineX
        diffY = spineTempY - spineY
        diffZ = spineTempZ - spineZ

        tempSum = 0

        for k in range(0, 20):
            listOne = linesOne[k].split(',')

            listTwo = lines[k + 20 + flag].split(',')

            tempSum += (((num(listTwo[0]) - num(diffX)) - num(listOne[0])) ** (2) + (
                (num(listTwo[1]) - num(diffY)) - num(listOne[1])) ** (2) + (
                            (num(listTwo[2]) - num(diffZ)) - num(listOne[2])) ** (2)) ** (0.5)

        # print tempSum

        j += 20
        flag += 20
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

# 2
probabilities_max_2 = []

for i in range(len(testing_list_2)):

    answer_testing_1 = []

    filename = file_prefix + str(testing_list_2[i]) + '.txt'
    print (filename)

    lines = open(filename).read().splitlines()
    print (lines[0])

    flag = 0
    print (len(lines))
    for j in range(10, (len(lines) - 29)):
        spineLineNumber = 30 + flag
        if spineLineNumber > len(lines):
            break

        spineLine = lines[spineLineNumber]
        spineLineArray = spineLine.split(',')

        spineTempX = num(spineLineArray[0])
        spineTempY = num(spineLineArray[1])
        spineTempZ = num(spineLineArray[2])

        diffX = spineTempX - spineX
        diffY = spineTempY - spineY
        diffZ = spineTempZ - spineZ

        tempSum = 0

        for k in range(0, 20):
            listOne = linesOne[k].split(',')

            listTwo = lines[k + 20 + flag].split(',')

            tempSum += (((num(listTwo[0]) - num(diffX)) - num(listOne[0])) ** (2) + (
                (num(listTwo[1]) - num(diffY)) - num(listOne[1])) ** (2) + (
                            (num(listTwo[2]) - num(diffZ)) - num(listOne[2])) ** (2)) ** (0.5)

        # print tempSum

        j += 20
        flag += 20
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

    probabilities_max_2.append(
        [test_probability_1, test_probability_2, test_probability_3, test_probability_4, test_probability_5,
         test_probability_6, test_probability_7, test_probability_8, test_probability_9, test_probability_10])

# print (linesOne)

# print (answer_testing_1)

# calculating accuracy

counter_probability = 0

for i in range(0, len(probabilities_max_2)):
    if max(probabilities_max_2[i]) == probabilities_max_2[i][1]:
        counter_probability += 1
    print(max(probabilities_max_2[i]))
    print(probabilities_max_2[i][1])

print('accuracy')
print(counter_probability)
print('-----')
print(len(probabilities_max_2))
print('==')
print(counter_probability / len(probabilities_max_2))

# 3
probabilities_max_3 = []

for i in range(len(testing_list_3)):

    answer_testing_1 = []

    filename = file_prefix + str(testing_list_3[i]) + '.txt'
    print (filename)

    lines = open(filename).read().splitlines()
    print (lines[0])

    flag = 0
    print (len(lines))
    for j in range(10, (len(lines) - 29)):
        spineLineNumber = 30 + flag
        if spineLineNumber > len(lines):
            break

        spineLine = lines[spineLineNumber]
        spineLineArray = spineLine.split(',')

        spineTempX = num(spineLineArray[0])
        spineTempY = num(spineLineArray[1])
        spineTempZ = num(spineLineArray[2])

        diffX = spineTempX - spineX
        diffY = spineTempY - spineY
        diffZ = spineTempZ - spineZ

        tempSum = 0

        for k in range(0, 20):
            listOne = linesOne[k].split(',')

            listTwo = lines[k + 20 + flag].split(',')

            tempSum += (((num(listTwo[0]) - num(diffX)) - num(listOne[0])) ** (2) + (
                (num(listTwo[1]) - num(diffY)) - num(listOne[1])) ** (2) + (
                            (num(listTwo[2]) - num(diffZ)) - num(listOne[2])) ** (2)) ** (0.5)

        # print tempSum

        j += 20
        flag += 20
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

    probabilities_max_3.append(
        [test_probability_1, test_probability_2, test_probability_3, test_probability_4, test_probability_5,
         test_probability_6, test_probability_7, test_probability_8, test_probability_9, test_probability_10])

# print (linesOne)

# print (answer_testing_1)

# calculating accuracy

counter_probability = 0

for i in range(0, len(probabilities_max_3)):
    if max(probabilities_max_3[i]) == probabilities_max_3[i][2]:
        counter_probability += 1
    print(max(probabilities_max_3[i]))
    print(probabilities_max_3[i][2])

print('accuracy')
print(counter_probability)
print('-----')
print(len(probabilities_max_3))
print('==')
print(counter_probability / len(probabilities_max_3))

# 4
probabilities_max_4 = []

for i in range(len(testing_list_4)):

    answer_testing_1 = []

    filename = file_prefix + str(testing_list_4[i]) + '.txt'
    print (filename)

    lines = open(filename).read().splitlines()
    print (lines[0])

    flag = 0
    print (len(lines))
    for j in range(10, (len(lines) - 29)):
        spineLineNumber = 30 + flag
        if spineLineNumber > len(lines):
            break

        spineLine = lines[spineLineNumber]
        spineLineArray = spineLine.split(',')

        spineTempX = num(spineLineArray[0])
        spineTempY = num(spineLineArray[1])
        spineTempZ = num(spineLineArray[2])

        diffX = spineTempX - spineX
        diffY = spineTempY - spineY
        diffZ = spineTempZ - spineZ

        tempSum = 0

        for k in range(0, 20):
            listOne = linesOne[k].split(',')

            listTwo = lines[k + 20 + flag].split(',')

            tempSum += (((num(listTwo[0]) - num(diffX)) - num(listOne[0])) ** (2) + (
                (num(listTwo[1]) - num(diffY)) - num(listOne[1])) ** (2) + (
                            (num(listTwo[2]) - num(diffZ)) - num(listOne[2])) ** (2)) ** (0.5)

        # print tempSum

        j += 20
        flag += 20
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

    probabilities_max_4.append(
        [test_probability_1, test_probability_2, test_probability_3, test_probability_4, test_probability_5,
         test_probability_6, test_probability_7, test_probability_8, test_probability_9, test_probability_10])

# print (linesOne)

# print (answer_testing_1)

# calculating accuracy

counter_probability = 0

for i in range(0, len(probabilities_max_4)):
    if max(probabilities_max_4[i]) == probabilities_max_4[i][3]:
        counter_probability += 1
    print(max(probabilities_max_4[i]))
    print(probabilities_max_4[i][3])

print('accuracy')
print(counter_probability)
print('-----')
print(len(probabilities_max_4))
print('==')
print(counter_probability / len(probabilities_max_4))

# 5
probabilities_max_5 = []

for i in range(len(testing_list_5)):

    answer_testing_1 = []

    filename = file_prefix + str(testing_list_5[i]) + '.txt'
    print (filename)

    lines = open(filename).read().splitlines()
    print (lines[0])

    flag = 0
    print (len(lines))
    for j in range(10, (len(lines) - 29)):
        spineLineNumber = 30 + flag
        if spineLineNumber > len(lines):
            break

        spineLine = lines[spineLineNumber]
        spineLineArray = spineLine.split(',')

        spineTempX = num(spineLineArray[0])
        spineTempY = num(spineLineArray[1])
        spineTempZ = num(spineLineArray[2])

        diffX = spineTempX - spineX
        diffY = spineTempY - spineY
        diffZ = spineTempZ - spineZ

        tempSum = 0

        for k in range(0, 20):
            listOne = linesOne[k].split(',')

            listTwo = lines[k + 20 + flag].split(',')

            tempSum += (((num(listTwo[0]) - num(diffX)) - num(listOne[0])) ** (2) + (
                (num(listTwo[1]) - num(diffY)) - num(listOne[1])) ** (2) + (
                            (num(listTwo[2]) - num(diffZ)) - num(listOne[2])) ** (2)) ** (0.5)

        # print tempSum

        j += 20
        flag += 20
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

    probabilities_max_5.append(
        [test_probability_1, test_probability_2, test_probability_3, test_probability_4, test_probability_5,
         test_probability_6, test_probability_7, test_probability_8, test_probability_9, test_probability_10])

# print (linesOne)

# print (answer_testing_1)

# calculating accuracy

counter_probability = 0

for i in range(0, len(probabilities_max_5)):
    if max(probabilities_max_5[i]) == probabilities_max_5[i][4]:
        counter_probability += 1
    print(max(probabilities_max_5[i]))
    print(probabilities_max_5[i][4])

print('accuracy')
print(counter_probability)
print('-----')
print(len(probabilities_max_5))
print('==')
print(counter_probability / len(probabilities_max_5))

# 6
probabilities_max_6 = []

for i in range(len(testing_list_6)):

    answer_testing_1 = []

    filename = file_prefix + str(testing_list_6[i]) + '.txt'
    print (filename)

    lines = open(filename).read().splitlines()
    print (lines[0])

    flag = 0
    print (len(lines))
    for j in range(10, (len(lines) - 29)):
        spineLineNumber = 30 + flag
        if spineLineNumber > len(lines):
            break

        spineLine = lines[spineLineNumber]
        spineLineArray = spineLine.split(',')

        spineTempX = num(spineLineArray[0])
        spineTempY = num(spineLineArray[1])
        spineTempZ = num(spineLineArray[2])

        diffX = spineTempX - spineX
        diffY = spineTempY - spineY
        diffZ = spineTempZ - spineZ

        tempSum = 0

        for k in range(0, 20):
            listOne = linesOne[k].split(',')

            listTwo = lines[k + 20 + flag].split(',')

            tempSum += (((num(listTwo[0]) - num(diffX)) - num(listOne[0])) ** (2) + (
                (num(listTwo[1]) - num(diffY)) - num(listOne[1])) ** (2) + (
                            (num(listTwo[2]) - num(diffZ)) - num(listOne[2])) ** (2)) ** (0.5)

        # print tempSum

        j += 20
        flag += 20
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

    probabilities_max_6.append(
        [test_probability_1, test_probability_2, test_probability_3, test_probability_4, test_probability_5,
         test_probability_6, test_probability_7, test_probability_8, test_probability_9, test_probability_10])

# print (linesOne)

# print (answer_testing_1)

# calculating accuracy

counter_probability = 0

for i in range(0, len(probabilities_max_6)):
    if max(probabilities_max_6[i]) == probabilities_max_6[i][5]:
        counter_probability += 1
    print(max(probabilities_max_6[i]))
    print(probabilities_max_6[i][5])

print('accuracy')
print(counter_probability)
print('-----')
print(len(probabilities_max_6))
print('==')
print(counter_probability / len(probabilities_max_6))

# 7
probabilities_max_7 = []

for i in range(len(testing_list_7)):

    answer_testing_1 = []

    filename = file_prefix + str(testing_list_7[i]) + '.txt'
    print (filename)

    lines = open(filename).read().splitlines()
    print (lines[0])

    flag = 0
    print (len(lines))
    for j in range(10, (len(lines) - 29)):
        spineLineNumber = 30 + flag
        if spineLineNumber > len(lines):
            break

        spineLine = lines[spineLineNumber]
        spineLineArray = spineLine.split(',')

        spineTempX = num(spineLineArray[0])
        spineTempY = num(spineLineArray[1])
        spineTempZ = num(spineLineArray[2])

        diffX = spineTempX - spineX
        diffY = spineTempY - spineY
        diffZ = spineTempZ - spineZ

        tempSum = 0

        for k in range(0, 20):
            listOne = linesOne[k].split(',')

            listTwo = lines[k + 20 + flag].split(',')

            tempSum += (((num(listTwo[0]) - num(diffX)) - num(listOne[0])) ** (2) + (
                (num(listTwo[1]) - num(diffY)) - num(listOne[1])) ** (2) + (
                            (num(listTwo[2]) - num(diffZ)) - num(listOne[2])) ** (2)) ** (0.5)

        # print tempSum

        j += 20
        flag += 20
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

    probabilities_max_7.append(
        [test_probability_1, test_probability_2, test_probability_3, test_probability_4, test_probability_5,
         test_probability_6, test_probability_7, test_probability_8, test_probability_9, test_probability_10])

# print (linesOne)

# print (answer_testing_1)

# calculating accuracy

counter_probability = 0

for i in range(0, len(probabilities_max_7)):
    if max(probabilities_max_7[i]) == probabilities_max_7[i][6]:
        counter_probability += 1
    print(max(probabilities_max_7[i]))
    print(probabilities_max_7[i][6])

print('accuracy')
print(counter_probability)
print('-----')
print(len(probabilities_max_7))
print('==')
print(counter_probability / len(probabilities_max_7))

# 8
probabilities_max_8 = []

for i in range(len(testing_list_8)):

    answer_testing_1 = []

    filename = file_prefix + str(testing_list_8[i]) + '.txt'
    print (filename)

    lines = open(filename).read().splitlines()
    print (lines[0])

    flag = 0
    print (len(lines))
    for j in range(10, (len(lines) - 29)):
        spineLineNumber = 30 + flag
        if spineLineNumber > len(lines):
            break

        spineLine = lines[spineLineNumber]
        spineLineArray = spineLine.split(',')

        spineTempX = num(spineLineArray[0])
        spineTempY = num(spineLineArray[1])
        spineTempZ = num(spineLineArray[2])

        diffX = spineTempX - spineX
        diffY = spineTempY - spineY
        diffZ = spineTempZ - spineZ

        tempSum = 0

        for k in range(0, 20):
            listOne = linesOne[k].split(',')

            listTwo = lines[k + 20 + flag].split(',')

            tempSum += (((num(listTwo[0]) - num(diffX)) - num(listOne[0])) ** (2) + (
                (num(listTwo[1]) - num(diffY)) - num(listOne[1])) ** (2) + (
                            (num(listTwo[2]) - num(diffZ)) - num(listOne[2])) ** (2)) ** (0.5)

        # print tempSum

        j += 20
        flag += 20
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

    probabilities_max_8.append(
        [test_probability_1, test_probability_2, test_probability_3, test_probability_4, test_probability_5,
         test_probability_6, test_probability_7, test_probability_8, test_probability_9, test_probability_10])

# print (linesOne)

# print (answer_testing_1)

# calculating accuracy

counter_probability = 0

for i in range(0, len(probabilities_max_8)):
    if max(probabilities_max_8[i]) == probabilities_max_8[i][7]:
        counter_probability += 1
    print(max(probabilities_max_8[i]))
    print(probabilities_max_8[i][7])

print('accuracy')
print(counter_probability)
print('-----')
print(len(probabilities_max_8))
print('==')
print(counter_probability / len(probabilities_max_8))

# 9
probabilities_max_9 = []

for i in range(len(testing_list_9)):

    answer_testing_1 = []

    filename = file_prefix + str(testing_list_9[i]) + '.txt'
    print (filename)

    lines = open(filename).read().splitlines()
    print (lines[0])

    flag = 0
    print (len(lines))
    for j in range(10, (len(lines) - 29)):
        spineLineNumber = 30 + flag
        if spineLineNumber > len(lines):
            break

        spineLine = lines[spineLineNumber]
        spineLineArray = spineLine.split(',')

        spineTempX = num(spineLineArray[0])
        spineTempY = num(spineLineArray[1])
        spineTempZ = num(spineLineArray[2])

        diffX = spineTempX - spineX
        diffY = spineTempY - spineY
        diffZ = spineTempZ - spineZ

        tempSum = 0

        for k in range(0, 20):
            listOne = linesOne[k].split(',')

            listTwo = lines[k + 20 + flag].split(',')

            tempSum += (((num(listTwo[0]) - num(diffX)) - num(listOne[0])) ** (2) + (
                (num(listTwo[1]) - num(diffY)) - num(listOne[1])) ** (2) + (
                            (num(listTwo[2]) - num(diffZ)) - num(listOne[2])) ** (2)) ** (0.5)

        # print tempSum

        j += 20
        flag += 20
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

    probabilities_max_9.append(
        [test_probability_1, test_probability_2, test_probability_3, test_probability_4, test_probability_5,
         test_probability_6, test_probability_7, test_probability_8, test_probability_9, test_probability_10])

# print (linesOne)

# print (answer_testing_1)

# calculating accuracy

counter_probability = 0

for i in range(0, len(probabilities_max_9)):
    if max(probabilities_max_9[i]) == probabilities_max_9[i][8]:
        counter_probability += 1
    print(max(probabilities_max_9[i]))
    print(probabilities_max_9[i][8])

print('accuracy')
print(counter_probability)
print('-----')
print(len(probabilities_max_9))
print('==')
print(counter_probability / len(probabilities_max_9))

# 10
probabilities_max_10 = []

for i in range(len(testing_list_10)):

    answer_testing_1 = []

    filename = file_prefix + str(testing_list_10[i]) + '.txt'
    print (filename)

    lines = open(filename).read().splitlines()
    print (lines[0])

    flag = 0
    print (len(lines))
    for j in range(10, (len(lines) - 29)):
        spineLineNumber = 30 + flag
        if spineLineNumber > len(lines):
            break

        spineLine = lines[spineLineNumber]
        spineLineArray = spineLine.split(',')

        spineTempX = num(spineLineArray[0])
        spineTempY = num(spineLineArray[1])
        spineTempZ = num(spineLineArray[2])

        diffX = spineTempX - spineX
        diffY = spineTempY - spineY
        diffZ = spineTempZ - spineZ

        tempSum = 0

        for k in range(0, 20):
            listOne = linesOne[k].split(',')

            listTwo = lines[k + 20 + flag].split(',')

            tempSum += (((num(listTwo[0]) - num(diffX)) - num(listOne[0])) ** (2) + (
                (num(listTwo[1]) - num(diffY)) - num(listOne[1])) ** (2) + (
                            (num(listTwo[2]) - num(diffZ)) - num(listOne[2])) ** (2)) ** (0.5)

        # print tempSum

        j += 20
        flag += 20
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

    probabilities_max_10.append(
        [test_probability_1, test_probability_2, test_probability_3, test_probability_4, test_probability_5,
         test_probability_6, test_probability_7, test_probability_8, test_probability_9, test_probability_10])

# print (linesOne)

# print (answer_testing_1)

# calculating accuracy

counter_probability = 0

for i in range(0, len(probabilities_max_10)):
    if max(probabilities_max_10[i]) == probabilities_max_10[i][9]:
        counter_probability += 1
    print(max(probabilities_max_10[i]))
    print(probabilities_max_10[i][9])

print('accuracy')
print(counter_probability)
print('-----')
print(len(probabilities_max_10))
print('==')
print(counter_probability / len(probabilities_max_10))
