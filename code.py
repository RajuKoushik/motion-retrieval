from sklearn.cluster import KMeans
import numpy as np
from numpy import array
import collections


def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


filename = 'dataset.txt'
lines = open(filename).read().splitlines()
print (lines[0])

dad = lines[3]
print (dad)

tat = dad.split(' ')

spineX = num(tat[1])
spineY = num(tat[2])
spineZ = num(tat[3])

answe = [] * 100000

totalDistance = 0
flag = 0
l = 0

for x in range(28, 7392):
    spineLineNumber = 31 + flag
    if spineLineNumber > 7392:
        break
    spineLine = lines[spineLineNumber]
    spineLineArray = spineLine.split(' ')
    spineTempX = num(spineLineArray[1])
    spineTempY = spineLineArray[2]
    spineTempZ = spineLineArray[3]

    diffX = spineTempX - spineX
    diffY = num(spineTempY) - num(spineY)
    diffZ = num(spineTempZ) - num(spineZ)

    tempSum = 0

    for i in range(3, 29):
        listOne = lines[i].split(' ')
        if (i + 28 + flag) > 7392:
            break
        listTwo = lines[i + 28 + flag].split(' ')
        tempSum = tempSum + (((num(listTwo[1]) - num(diffX)) - num(listOne[1])) ** (2) + (
        (num(listTwo[2]) - num(diffY)) - num(listOne[2])) ** (2) + (
                             (num(listTwo[3]) - num(diffZ)) - num(listOne[3])) ** (2)) ** (0.5)

    answe.append(tempSum)
    print (tempSum)

    l = l + 1

    flag = flag + 28
    x = x + 28
print (answe)



numpyArray = array(answe)

numArray= numpyArray.reshape(-1,1)

kmeans = KMeans(n_clusters=10, random_state=0).fit(numArray)

print (kmeans.labels_)
label = kmeans.labels_
label_list = label.tolist()
print(len(label_list))

#here's an interesting information
counter=collections.Counter(label_list)
print(counter)

color = np.random.rand(cluster_num)








print (kmeans.cluster_centers_)
