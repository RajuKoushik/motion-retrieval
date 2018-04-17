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

linesOne = [0] * 100

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

        coordinateX = [] * 1000
        coordinateY = [] * 1000
        coordinateZ = [] * 1000

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

        answer.append(tempSum)
# print (answer)
print (linesOne)

numpyArray = array(answer)

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

walk_model = [[0] * 100 for i in range(100)]

for l in range(1, int(counter_walk)):
    walk_model[kmeans.labels_[l]][kmeans.labels_[l - 1]] += 1

print (walk_model)

total_sum = total_sum + int(counter_walk)

# grab model

sum_grab = 0

for p1 in range(0, len(random_list)):
    if (random_list[p1] >= 40 and random_list[p1] < 80):
        filename = file_prefix + str(random_list[p]) + '.txt'
        print (filename)

        lines = open(filename).read().splitlines()
        print (len(lines))
        sub_length_grab = len(lines)
        sum_grab += sub_length_grab
    if (random_list[p1] >= 80):
        break

counter_grab = sum_grab / 20

grab_model = [[0] * 100 for i in range(100)]

for l in range(total_sum, total_sum + int(counter_grab)):
    grab_model[kmeans.labels_[l]][kmeans.labels_[l - 1]] += 1

print (grab_model)

total_sum = total_sum + int(counter_grab)

# watch clock model

sum_watch = 0

for p1 in range(0, len(random_list)):
    if (random_list[p1] >= 80 and random_list[p1] < 120):
        filename = file_prefix + str(random_list[p]) + '.txt'
        print (filename)

        lines = open(filename).read().splitlines()
        print (len(lines))
        sub_length_watch = len(lines)
        sum_watch += sub_length_watch
    if (random_list[p] >= 120):
        break

counter_watch = sum_watch / 20

watch_model = [[0] * 100 for i in range(100)]

for l in range(total_sum, total_sum + int(counter_watch)):
    watch_model[kmeans.labels_[l]][kmeans.labels_[l - 1]] += 1

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
    if (random_list[p1] >= 160):
        break

counter_head = sum_head / 20

head_model = [[0] * 100 for i in range(100)]

for l in range(total_sum, total_sum + int(counter_head)):
    head_model[kmeans.labels_[l]][kmeans.labels_[l - 1]] += 1

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
    if (random_list[p1] >= 200):
        break

counter_phone = sum_phone / 20

phone_model = [[0] * 100 for i in range(100)]

for l in range(total_sum, total_sum + int(counter_phone)):
    phone_model[kmeans.labels_[l]][kmeans.labels_[l - 1]] += 1

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
    if (random_list[p1] >= 240):
        break

counter_arms = sum_arms / 20

arms_model = [[0] * 100 for i in range(100)]

for l in range(total_sum, total_sum + int(counter_arms)):
    arms_model[kmeans.labels_[l]][kmeans.labels_[l - 1]] += 1

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
    if (random_list[p1] >= 280):
        break

counter_seat = sum_seat / 20

seat_model = [[0] * 100 for i in range(100)]

for l in range(total_sum, total_sum + int(counter_seat)):
    seat_model[kmeans.labels_[l]][kmeans.labels_[l - 1]] += 1

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
    if (random_list[p1] >= 320):
        break

counter_punch = sum_punch / 20

punch_model = [[0] * 100 for i in range(100)]

for l in range(total_sum, total_sum + int(counter_punch)):
    punch_model[kmeans.labels_[l]][kmeans.labels_[l - 1]] += 1

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
    if (random_list[p1] >= 360):
        break

counter_kick = sum_kick / 20

kick_model = [[0] * 100 for i in range(100)]

for l in range(total_sum, total_sum + int(counter_kick)):
    kick_model[kmeans.labels_[l]][kmeans.labels_[l - 1]] += 1

print (kick_model)

total_sum = total_sum + int(counter_kick)

# wave

sum_wave = 0

for p in range(0, len(random_list)):
    if (random_list[p] >= 360 and random_list[p] < 400):
        filename = file_prefix + str(random_list[p]) + '.txt'
        print (filename)

        lines = open(filename).read().splitlines()
        print (len(lines))
        sub_length_wave = len(lines)
        sum_wave += sub_length_wave

counter_wave = sum_wave / 20

wave_model = [[0] * 100 for i in range(100)]

for l in range(total_sum, total_sum + int(counter_wave)):
    wave_model[kmeans.labels_[l]][kmeans.labels_[l - 1]] += 1

print (wave_model)

total_sum = total_sum + int(counter_wave)
