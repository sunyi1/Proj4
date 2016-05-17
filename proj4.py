import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys

numberOfCluster = sys.argv[1]
inputFile = sys.argv[2]
readFile = open(inputFile, "r")
wholeFile = readFile.readlines()


X = []
Y = []
list1= []
for line in wholeFile:
    #print(line)
    #line.strip( )
    #list=[line.rstrip('\n')]
    #number.append(list)
    list1= line.rstrip('\n').split(',')
    X.append(float(list1[0]))
    Y.append(float(list1[1]))

Z = np.array( list(zip(X,Y)) )


Z = np.float32(Z)

# kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv2.kmeans(Z,int(numberOfCluster),None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)


A = Z[label.ravel()==0]
B = Z[label.ravel()==1]
C = Z[label.ravel()==2]
D = Z[label.ravel()==3]
# Plot all data
plt.scatter(A[:,0],A[:,1])
plt.scatter(B[:,0],B[:,1],c = 'r')
plt.scatter(C[:,0],C[:,1],c = 'r')
plt.scatter(D[:,0],D[:,1],c = 'r')
plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
plt.show()