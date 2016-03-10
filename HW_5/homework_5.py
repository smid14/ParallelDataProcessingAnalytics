from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS
from numpy import array
import os
import sys

# Set the path for spark installation
# this is the path where you downloaded and uncompressed the Spark download
# Using forward slashes on windows, \\ should work too.
os.environ['SPARK_HOME'] = "/home/manuel/Dokumente/spark-1.6.0"
# Append the python dir to PYTHONPATH so that pyspark could be found
sys.path.append("/home/manuel/Dokumente/spark-1.6.0/python")
# Append the python/build to PYTHONPATH so that py4j could be found
sys.path.append('/home/manuel/Dokumente/spark-1.6.0/python/lib/py4j-0.9-src.zip')



if __name__ == "__main__":

    sc = SparkContext(appName="PS 7")
    # Load and parse the data
    data = sc.textFile("data/mllib/als/test.data")
    ratings = data.map(lambda line: array([float(x) for x in line.split(',')]))

    # Build the recommendation model using Alternating Least Squares
    rank = 10
    numIterations = 20
    model = ALS.train(ratings, rank, numIterations)

    # Evaluate the model on training data
    testdata = ratings.map(lambda p: (int(p[0]), int(p[1])))
    predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
    ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
    MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).reduce(lambda x, y: x + y)/ratesAndPreds.count()
    print("Mean Squared Error = " + str(MSE))