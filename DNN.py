# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 13:16:27 2019

@author: TsungYuan
"""

from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Flatten
from keras.optimizers import  Adam
import csv

def load_data():
    trainX = np.load('trainX_27000_10.npy')
    testX = np.load('private_27000_10.npy')
    label = np.load('label8.npy')
    return trainX, testX, label

def build_model():
        model = Sequential()
        model.add(Dense(input_shape=(10,100,),units=1000,activation='relu'))
        model.add(Dense(units=500,activation='relu'))
        model.add(Dense(units=250,activation='relu'))
        model.add(Dense(units=150,activation='relu'))
        model.add(Dense(units=50,activation='relu'))
        model.add(Dense(units=16,activation='relu'))
        model.add(Flatten())
        model.add(Dense(units=8,activation='softmax'))
        model.summary()
        return model

trainX, testX, label = load_data()
model = build_model()

model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
model.fit(trainX,label[0:7000],batch_size=50,epochs=5,validation_split=0.2)
score = model.evaluate(trainX,label[0:7000])
print ('\nTrain Acc:', score[1])
preds = model.predict_classes(testX)

results = np.zeros((20000,4)) 
for i in range(20000):
    if preds[i] == 0:
        results[i,2] = 1
    elif preds[i] == 1:
        results[i,1] = 1 
    elif preds[i] == 2:
        results[i,1] = 1 
        results[i,2] = 1 
    elif preds[i] == 3:
        results[i,3] = 1 
    elif preds[i] == 4:
        results[i,0] = 1 
    elif preds[i] == 5:
        results[i,0] = 1 
        results[i,2] = 1 
    elif preds[i] == 6:
        results[i,0] = 1 
        results[i,1] = 1 
    elif preds[i] == 7:
        results[i,0] = 1 
        results[i,1] = 1 
        results[i,2] = 1 

with open('output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(results)
