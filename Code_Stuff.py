import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
#from numpy import array
import numpy as np



points = [[1, 1], [3, 2], [6, 3], [8, 4]]
print ("Data Set")
points_mat = np.matrix(points)
print (points_mat)


print ("Condensed distance matrix")
dist_mat = pdist(points_mat)
print (dist_mat)

print ("Average Linkage")
linkage_matrix = linkage(dist_mat, 'average')
print (linkage_matrix)

plt.figure("Avg. Linkage")
plt.title("Avg. Linkage")
dendrogram(linkage_matrix)
plt.show()