__author__ = 'developer'
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import sys
import os
import numpy as np
import random



# Set the path for spark installation
# this is the path where you downloaded and uncompressed the Spark download
# Using forward slashes on windows, \\ should work too.
os.environ['SPARK_HOME'] = "/home/manuel/Dokumente/spark-1.6.0"
# Append the python dir to PYTHONPATH so that pyspark could be found
sys.path.append("/home/manuel/Dokumente/spark-1.6.0/python")
# Append the python/build to PYTHONPATH so that py4j could be found
sys.path.append('/home/manuel/Dokumente/spark-1.6.0/python/lib/py4j-0.9-src.zip')

from pyspark import SparkContext


"""
Method given by the University to parse the vector of numbers
"""
def parseVector(line):
    return np.array([float(x) for x in line.split(' ')])




"""
Method given by the University to calculate the distance of a point (p) to all centroids (list:centers)
and returns the index of the nearest centroid
"""
def closestPoint(p, centers):
    bestIndex = 0
    closest = float("+inf")    # floating point infinity literal
    for i in range(len(centers)):   #iterate over all centroids
        tempDist = np.sum((p - centers[i]) ** 2) #calculates the distance of a set of points
        if tempDist < closest:                   #if the tempdistance is smaller than the actual distance
            closest = tempDist                   #set tempdistance as actual distance
            bestIndex = i                        #save index i (for loop) as best index
    return bestIndex


def distanceCentroidsMoved(oldCentroids, newCentroids):
    """
    Calculates the sum of the shift of the old and new centroids
    :return: the float of the change
    """
    sum = 0.0
    for index in range(len(oldCentroids)):
        sum += np.sum((oldCentroids[index] - newCentroids[index]) ** 2)
    return sum


"""
Method given by the University to generate a dataset in the 2 dimensional space
"""
def generateData(N, k):
    """ Generates N 2D points in k clusters
        From http://datasciencelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python/
    """
    n = float(N) / k
    X = []
    for i in range(k):
        c = (random.uniform(-1, 1), random.uniform(-1, 1))
        s = random.uniform(0.05, 0.5)
        x = []
        while len(x) < n:
            a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
            # Continue drawing points from the distribution in the range [-1,1]
            if abs(a) < 1 and abs(b) < 1:
                x.append([a, b])
        X.extend(x)
    X = np.array(X)[:N]
    # Write list to file for later use
    f = open('./dataset_N' + str(N) + '_K' + str(k) + '.txt', 'w')
    for x in X:
        f.write(str(x[0]) + " " + str(x[1]) + '\n')

    f.close();

    return X



def centroidsChanged(old, new):
    """

    :param old:
    :param new:
    :return:
    """
    if old is None or new is None:
        return True
    contained = False
    for p in new:
        contained = any((p == x).all() for x in old)
        if not contained:
            return True

    return False


def within_cluster_variation(closest, centroids):
    """
    The method calculates the Within Cluster Variation of the K-Means output.
    The method measures the sum of the squared distance between the single of a cluster to the centroid (cluster center)
    :param closest: RDD
    :param centroids: list()
    :return: None
    """

    #Check if Input is None
    if centroids is None:
        print ("The centroids are None!")
    if closest is None:
        print ("The closest RDD is None!")


    # Calculate the squared Distance to the centroid. Remember that closest is of the following form:
    # (centroid_index, (array([x,y]), 1)) where the 1 is the counter variable and hence point[0] points to the [x,y] variables in the array
    squaredDistance = closest.map(lambda (centroid_index, point): np.sum((centroids[centroid_index] - point[0]) ** 2))

    # Calculate the sum of the squaredDistance RDD with reduce
    withinClusterVariation = squaredDistance.reduce(lambda x,y: x+y)

    print ("The Within Cluster Variation is: ", withinClusterVariation)





if __name__ == "__main__":
    # Checks the Input Parameters
    if len(sys.argv) != 5:
        print (sys.stderr, "Usage: kmeans <file> <maxIterations> <Npoints> <k>")
        exit(-1)

    # Set Spark
    sc = SparkContext(appName="Python-K-Means")

    # Reads out the parameters
    fname = sys.argv[1]
    maxIterations = int(sys.argv[2])
    Npoints = int(sys.argv[3])
    K = int(sys.argv[4])

    # Generates new Data
    if fname is "" or not os.path.isfile(fname):
        fname = './dataset_N' + str(Npoints) + '_K' + str(K) + '.txt'
    data = None
    if os.path.isfile(fname):
        print("Loading data from file: " + fname)
        lines = sc.textFile(fname)
        data = lines.map(parseVector).cache()
    else:
        print("Generating new data for n=" + str(Npoints) + " and k=" + str(K))
        dataLocal = generateData(Npoints, K)
        data = sc.parallelize(dataLocal)

    print ("Number of points: %d" % (data.count()))
    print ("K: %d" % (K))

    print ("Data is: ", data.collect())

    # Samples the initial centroids and returns a list of tuples of the centroids.
    centroids = data.takeSample(False, K, 1)
    print ("Centroids: ", centroids)
    # At the beginning we have no newCentroids, afterwards we will calculate newCentroids
    newCentroids = None
    pointsToCluster = None
    counter = 0



    #Iterate through as long as the counter is less than the maxIterations or the centroids do not change any more
    while ((counter < maxIterations) and centroidsChanged(centroids, newCentroids)):


        print ("\n Iteration: %d" % (counter))
        # Increase the counter by one
        counter += 1


        # Test if we are in the first iteration:
        # If not, then save the newCentroids as the 'old' centroids
        if None != newCentroids:
            centroids = newCentroids[:]
        # If we are, then set the newCentroids to the random Draw centroids from before
        else:
            newCentroids = centroids[:]


        # Set the datapoints into the corresponding cluster with the minimum distance to the cluster centroids. Remember that centroids is a k-length list of points!
        # The (p,1) where p = (x,y) tupel helps to count the number of points in a cluster in the following loop
        pointsToCluster = data.map(lambda p: (closestPoint(p, centroids), (p, 1)))
        #print ("Closest Points are: ",closest.collect())

        # Iteration over each Cluster
        for i in range(K):

            # Take the points of each cluster in a iteration
            eachCluster = pointsToCluster.filter(lambda x: x[0] == i).map(lambda x: x[1])   # Filtert die einzelnen cluster raus
            #print ("ClosestOneCluster: ", closestOneCluster.collect())
            #print ("Cluster %d has %d data points" % (cIndex, closestOneCluster.count()))

            # Calculate the sum of (x,y) values of points in the cluster by point1+point2 and the occurence of points in each cluster
            # Result is an datastructure of (array[x,y],count)

            coordSumAndPointCount = eachCluster.reduce(lambda (point1, count1), (point2, count2): (point1 + point2, count1 + count2))
            #print ("SumAndCountOneCluster: ", sumAndCountOneCluster)

            #The sum of the (x,y) coordinates of all points in the cluster
            coordSum = coordSumAndPointCount[0]

            #The count of points in the cluster
            pointCount = coordSumAndPointCount[1]


            #Set the new Centroids for each cluster
            newCentroids[i] = coordSum / pointCount

        tempDist = distanceCentroidsMoved(centroids, newCentroids)

    print ("------Final centroids of K-Means are------: ",centroids)

    within_cluster_variation(pointsToCluster,centroids)

    sc.stop()
