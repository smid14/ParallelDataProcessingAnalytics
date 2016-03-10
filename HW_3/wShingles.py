'''
The Program implements the w-Shingles of a Text and compares two w-Shingling Sets by the Jaccard Similarity Measure
The two main methods are the wShingling(text, w) method and the loop(base, sum, i) method that uses
a recursion to add the single characters of a text to a w-shingle
'''


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




def compareText(text1,text2, w):
    '''
    Method that computes the shingles for the two input texts for a given w
    and computes the similarity between the two text sets
    :param text1: str
    :param text2: str
    :param w: int
    :return: None
    '''
    setText1 = wShingles(text1, w)
    setText2 = wShingles(text2, w)
    setIntersection = setText1.intersection(setText2)
    num = setIntersection.count()
    setUnion = setText1.union(setText2).distinct()
    den = setUnion.count()
    jaccardSimilarity = float(num)/den
    print ("Similarity of the two Sets is: ", jaccardSimilarity)



def wShingles(text, w):
    '''
    Method reads in the text str as an Spark RDD and maps to each char an index.
    :param text: str
    :param w: int
    :return:
    '''
    chrRdd = sc.parallelize(text)
    indexRdd = chrRdd.zipWithIndex().map(lambda (x,y): (y,x))
    #print (indexRdd.collect())
    return loop(indexRdd,indexRdd,w)


def loop(base, sum, i):
    '''
    Recursive Method to sum/add up the char to a size of i/w.
    :param base: str
    :param sum: str
    :param i: int
    :return:
    '''
    if (i<=1):
      return sum.map(lambda (x,y): y)
    else:
      segment = base.map(lambda (id, seq): (id-1,seq))
      #print ("Sum: ", sum.collect())
      #print ("Segment: ",segment.collect())
      return loop(segment, sum.join(segment).map(lambda (k, (seq1, seq2)): (k, seq1+seq2)), i-1)



if __name__ == "__main__":
    sc = SparkContext(appName="W-Shingles")
    text1 = "The legal system is made up of civil courts, criminal courts and specialty courts such as family law courts and bankruptcy court. Each court has its own jurisdiction, which refers to the cases that the court is allowed to hear. In some instances, a case can only be heard in one type of court. For example, a bankruptcy case must be heard in a bankruptcy court. In other instances, there may be several potential courts with jurisdiction. For example, a federal criminal court and a state criminal court would each have jurisdiction over a crime that is a federal drug offense but that is also an offense on the state level."
    text2 = "The legal system is comprised of criminal and civil courts and specialty courts like bankruptcy and family law courts. Every one of the courts is vested with its own jurisdiction. Jurisdiction means the types of cases each court is permitted to rule on. Sometimes, only one type of court can hear a particular case. For instance, bankruptcy cases an be ruled on only in bankruptcy court. In other situations, it is possible for more than one court to have jurisdiction. For instance, both a state and federal criminal court could have authority over a criminal case that is illegal under federal and state drug laws."
    compareText(text1, text2, 9)
