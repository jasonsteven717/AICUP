# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:56:10 2019

@author: TsungYuan
"""

from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import csv


trainX = np.load('trainX_27000.npy')
testX = np.load('testX_27000.npy')
label = np.load('label.npy')
trainX = trainX.reshape((7000,500))
testX = testX.reshape((20000,500))

knn = KNeighborsClassifier()
knn.fit(trainX,label)
test = knn.predict(testX)
with open('output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(test)
