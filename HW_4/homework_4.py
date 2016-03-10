'''

'''



from math import pow,sqrt
import os
import sys
import numpy as np
import re


# Set the path for spark installation
# this is the path where you downloaded and uncompressed the Spark download
# Using forward slashes on windows, \\ should work too.
os.environ['SPARK_HOME'] = "/home/manuel/Dokumente/spark-1.6.0"
# Append the python dir to PYTHONPATH so that pyspark could be found
sys.path.append("/home/manuel/Dokumente/spark-1.6.0/python")
# Append the python/build to PYTHONPATH so that py4j could be found
sys.path.append('/home/manuel/Dokumente/spark-1.6.0/python/lib/py4j-0.9-src.zip')



from pyspark import SparkContext


def deletePunctuation(text):
    text=text.lower().strip()
    text=re.sub('[^0-9a-zA-Z ]','', text)
    return text

def mostFrequent(list):
    freq = 0
    for element in list:
        if element[1][0] > freq:
            freq = element[1][0]
            word = element[1][1]
    return word, freq


def longestword(list):
    max = 0
    for e in list:
        if len(e[1]) > max:
            max = len(e[1])
            word = e[1]
    return word, max

if __name__ == "__main__":
    sc = SparkContext(appName="Problem Set 10")

    #Read the text file and make the word count of every word, sum and Avg of Word Count
    file = sc.textFile("pg44016.txt").map(deletePunctuation)
    print ("File: ", file.collect())
    counts = file.flatMap(lambda line: line.split(" ")) \
                .map(lambda word: (word, 1)) \
                .reduceByKey(lambda a, b: a + b)
    sum = counts.reduce(lambda (x1,y1),(x2,y2): ('Sum',y1+y2))
    amountWords = len(counts.collect())
    avg = float(sum[1])/amountWords


    #Calculate the Standard Deviation of the word distribution
    varianceNum = counts.map(lambda (x1,y1): (x1,pow(y1-avg,2)))
    varianceSum = varianceNum.reduce(lambda (x1,y1),(x2,y2): ('Total Std: ', y1+y2))
    variance = varianceSum[1]/amountWords
    standardDev = sqrt(variance)
    print 'The total Standard Deviation is: ', standardDev

    #Calculates the longest word among the top 5 words and the most frequent word among the 20 shortest words
    most_frequent = counts.map(lambda (x,y):(y,x)).sortByKey(False,1).top(5)
    word, freq = longestword(most_frequent)
    print 'The longest word among the top 5 words is: ', word , 'with a length of ', freq

    shortest_words = counts.map(lambda (x,y): (len(x), (y, x))).sortByKey(True).take(10)
    word, freq = mostFrequent(shortest_words)
    print 'The most frequent word among the 20 shortest words is the word: ',word, 'with a frequency of: ', freq