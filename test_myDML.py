# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 24:00:00 2017

@author: Zhiyu
"""
# import your module here
import myDML
import pickle as pkl
import numpy as np
import collections
import sys

# (global) variable definition here

# class definition here

# function definition here

# main program here
if  __name__ == '__main__':
    # test myDML.py
    # load two-moons-dataset
    train_file = open('dataset/moons/moons_train.pkl', 'rb')
    test_file = open('dataset/moons/moons_test.pkl', 'rb')
    moons_train = pkl.load(train_file) # tuple of training data
    moons_test = pkl.load(test_file) # tuple of testing data
    train_file.close()
    test_file.close()
    
    train_X = moons_train[0] # instances of training data
    train_Y = moons_train[1] # labels of training data
    test_X = moons_test[0] # instances of testing data
    test_Y = moons_test[1] # labels of testing data
    
    # training
    try:
        myDML.train(moons_train)
    except Exception as e:
        print(e)
        sys.exit(-1)
    
    # evaluating
    for K in [1, 3, 5]:
        # testing with Euclidean_distance
        predict_label = np.zeros(test_X.shape[0])
        for i in range(test_X.shape[0]):
            distance_vector = np.zeros(train_X.shape[0])
            for j in range(train_X.shape[0]):
                distance_vector[j] = myDML.Euclidean_distance(test_X[i], train_X[j])
            labels_of_K_neighbor = train_Y[distance_vector.argsort()[0:K]]
            predict_label[i] = collections.Counter(labels_of_K_neighbor).most_common(n=1)[0][0]
        test_error = np.sum(predict_label != test_Y)/test_Y.shape[0]
        print('Euclidean_distance+knn(k='+str(K)+'): \t%f' % test_error)
        
        # testing with your distance
        predict_label = np.zeros(test_X.shape[0])
        for i in range(test_X.shape[0]):
            distance_vector = np.zeros(train_X.shape[0])
            for j in range(train_X.shape[0]):
                distance_vector[j] = myDML.distance(test_X[i], train_X[j])
            labels_of_K_neighbor = train_Y[distance_vector.argsort()[0:K]]
            predict_label[i] = collections.Counter(labels_of_K_neighbor).most_common(n=1)[0][0]
        test_error = np.sum(predict_label != test_Y)/test_Y.shape[0]
        print('myDML.distance()+knn(k='+str(K)+'): \t%f' % test_error)