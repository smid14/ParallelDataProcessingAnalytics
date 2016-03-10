"""
First Homework
Goal: Get familiar with basic Spark commands. No specific goal
"""
import os
import sys
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



if __name__ == "__main__":
    sc = SparkContext(appName='homework_1')

    #create RDD:
    rdd = sc.parallelize([1,2,3,4,5,6,7,8,9])

    #Actions:
    #with collect() the RDD can be printed
    print ("Action - collect(): ",rdd.collect())

    #Show the first k = 5 elements with take()
    print ("Action - take(5): ",rdd.take(5))

    #Count the elements in the rdd with count()
    print ("Action - count(): ",rdd.count())

    #Sum the elements in the rdd with sum()
    print ("Action - sum(): ",rdd.sum())

    #Counts the occurence of a number in the rdd, countByValue()
    print ("Action - countByValue(): ",rdd.countByValue())



    #Transformation:
    #Union of two RDDs with union()
    x = sc.parallelize([("a",1),("b",2),("c",3)])
    y = sc.parallelize([("d",4),("e",5),("f",6)])
    print ("Transformation - union(): ", x.union(y).collect())

    #Distinct returns the distinct() elements in the RDD
    x = sc.parallelize([1,1,1,1,2,2,2,3,4,5,6,8,8,8,8,9,10,1,1,1])
    print ("Transformation - distinct(): ", x.distinct().collect())

    #Glom returns an RDD created by coalescing all elements within each partition into a list. Divides the
    #the x RDD into 3 RDDs.
    x = sc.parallelize([1,2,3,4,5,6,7,8,9,10],3)
    print ("Transformation - glom(): ", x.glom().collect())


    #JOIN
    #Joins the x RDD to the y RDD
    x = sc.parallelize([("a",1), ("b",4), ("c",10), ("d",100)])
    y = sc.parallelize([("a",2), ("a",8), ("e",5), ("c",100)])
    print ("Join x on y - join(): ", x.join(y).collect())
    #Result:  [('a', (1, 2)), ('a', (1, 8)), ('c', (10, 100))]


    #SORTBY:
    #Sort the tuple in the RDD for the first or second element
    x = sc.parallelize([("b",4), ("a",100),("c",10),("d",1), ("e", 50)])
    print ("Sort the first element in the tuple - sortBy(): ", x.sortBy(lambda x: x[0]).collect())
    #Return: [('a', 100), ('b', 4), ('c', 10), ('d', 1), ('e', 50)]
    print ("Sort the second element in the tuple - sortBy(): ", x.sortBy(lambda x: x[1]).collect())
    #Return: [('d', 1), ('b', 4), ('c', 10), ('e', 50), ('a', 100)]


    #GroupBy:
    #Groups the elements in the RDD into a False and True group if the rule is matched or not.
    data = sc.parallelize([1,1,2,3,45,6,6,7,8,9])
    res  = data.groupBy(lambda x: x>6)
    print ("Show the False and True group - groupBy: ", res)
    print ("Show the sorted Results: ", str([(x, sorted(y)) for (x,y) in res.collect()]))
    #Return: [(False, [1, 1, 2, 3, 6, 6]), (True, [7, 8, 9, 45])]

    #Reduce:
    data = sc.parallelize([1,1,22,3,4,45,12,1])
    res = data.reduce(lambda x,y: x+y)
    print ("Reduce 1: ", res)
    # ('Reduce 1: ', 89)


    data = sc.parallelize([(1,(1,12)), (1,(5,7)), (1, (10,8)), (1, (12,12))])
    res  = data.reduce(lambda (x1, y1), (x2, y2): (x1+x2,(y1[0]+y2[0],y1[1]+y2[1])))
    print ("Reduce 2: ", res)
    # ('Reduce 2: ', (4, (28, 39)))

    #ReduceByKey
    data = sc.parallelize([("a",(1,12)), ("b",(5,7)), ("c", (10,8)), ("a", (12,12))])
    res = data.reduceByKey(lambda x,y: (x[0]+y[0],x[1]+y[1]))
    print ("ReduceByKey: ", res.collect())
    # ('ReduceByKey: ', [('a', (13, 24)), ('c', (10, 8)), ('b', (5, 7))])

    #ZipWithIndex:
    data = sc.parallelize([(2,3), (4,1), (1,3), (5,1)])
    res = data.zipWithIndex()
    print ("ZipWithIndex: ", res.collect())

    inClusterList = res.map(lambda (point, index): (point, (point, 1), index))

    for e in inClusterList.collect():
        point, point_counter, index = e
        print ("Point: ", point, "Point Counter: ", point_counter, "Index: ", index)
        pDistance = inClusterList.map(lambda (p, c, ip): (np.sum((p - point)**2), (c,point_counter), (ip, index)))
        print ("P_Distance: ", pDistance)

    #dataPoints = generateData(5,3)
    data = sc.parallelize([(1,1), (3,2),(6,3),(8,4)])
    clusterList = data.zipWithIndex()

    print (clusterList.collect())
    distance = sc.parallelize([])
    rdd_list = clusterList.collect()
    print ("Rdd List: ",rdd_list)
    print ("First Rdd List: ", rdd_list[0])
    length = len(rdd_list)
    print (length)

    combined_cluster = clusterList.cartesian(clusterList)
    print ("Combined Cluster List: ", combined_cluster.collect())
    combination_cluster = combined_cluster.filter(lambda (s1, s2): s1[0] < s2[0])
    print ("Combination cluster: ", combination_cluster.collect())
    combination_cluster = combination_cluster.map(lambda (p_1, p_2): ((p_1[0],p_2[0]), (p_1[1], p_2[1])))
    print ("Combination cluster New: ", combination_cluster.collect())
    partialDistance = combination_cluster.map(lambda (((p),(q)), (indexes)): (np.sum((np.array(p)-np.array(q))**2), indexes))
    print ("Partial Distances: ", partialDistance.collect())
    combination_list = combination_cluster.collect()

