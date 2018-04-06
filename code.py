from sklearn.cluster import KMeans
import numpy as np
from numpy import array
import collections
import matplotlib.pyplot as plt
import sys

sys.path.append("../tools/")
import pylab as pl

import plotly.plotly as py
import plotly.graph_objs as go


def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color=colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()


def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


filename = 'dataset.txt'
lines = open(filename).read().splitlines()
print (lines[0])

# scraping the dataset
fla = 0
bigX = [] * 100000

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

    # scraping baby
    coordinateX = [] * 1000
    coordinateY = [] * 1000
    coordinateZ = [] * 1000

    for i in range(3, 29):

        listOne = lines[i].split(' ')
        if (i + 28 + flag) > 7392:
            break
        coordinateX.append(num(listOne[1]))
        coordinateY.append(num(listOne[2]))
        coordinateZ.append(num(listOne[3]))

        listTwo = lines[i + 28 + flag].split(' ')
        tempSum = tempSum + (((num(listTwo[1]) - num(diffX)) - num(listOne[1])) ** (2) + (
            (num(listTwo[2]) - num(diffY)) - num(listOne[2])) ** (2) + (
                                 (num(listTwo[3]) - num(diffZ)) - num(listOne[3])) ** (2)) ** (0.5)

    bigX.append([coordinateX,coordinateY,coordinateZ])

    answe.append(tempSum)
    print (tempSum)

    l = l + 1

    flag = flag + 28
    x = x + 28
print (answe)

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure()
ax = Axes3D(fig)



ax = plt.axes(projection='3d')
zline = array(bigX[0][0])
xline = array(bigX[0][1])
yline = array(bigX[0][2])

ax.scatter(xs=zline[:-1], ys=xline[:-1], zs=yline[:-1], zdir='z', label='ys=0, zdir=z')
plt.show()



print(zline[:-1])
print(xline[:-1])
print(yline[:-1])

ax.plot3D(xline, yline, zline, 'gray')
plt.show()

numpyArray = array(answe)

numArray = numpyArray.reshape(-1, 1)

kmeans = KMeans(n_clusters=10, random_state=0).fit(numArray)

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

#pl.scatter(numArray[:, 0], numArray[:, 0], c=kmeans.labels_)
#pl.show()
