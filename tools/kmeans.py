from random import randint
import random

def kmeans(values, numOfCentroids, numOfIter, heigth, width):
    labeledValues = {p: 0 for p in values}
    if numOfCentroids<=1:
        return labeledValues

    centroids = []
    for i in range(numOfCentroids):
        centroids.append(values[random.choice(values.keys())])

    #print "random centroids: "+str(centroids)
    iteration = 0
    changed = True
    while iteration<=numOfIter and changed==True:
        #print "iteration: "+str(iteration)
        changed = False
        oldCentroids = centroids[:]
        for point in values:
            minC = 0
            minDist = abs(int(values[point]) - int(centroids[0]))

            for i in range(1,numOfCentroids):
                dist = abs(int(values[point]) - int(centroids[i]))
                if dist < minDist:
                    minDist = dist
                    minC = i

            labeledValues[point] = centroids[minC]


        #print labeledValues
        #print centroids
        for i in range(numOfCentroids):
            c = centroids[i]
            sumOfValues = 0.0
            numOfValues = 0.0
            for point in values:
                if labeledValues[point] == c:
                    sumOfValues += values[point]*1.0
                    numOfValues += 1.0

            if numOfValues!=0:
                centroids[i] = sumOfValues/numOfValues
            #print abs(centroids[i] - oldCentroids[i])
            if abs(centroids[i] - oldCentroids[i])>0.001:
                changed = True
        #print centroids
        iteration += 1
        #print changed
    #print centroids
    #return labeledValues
    return centroids