# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 24:00:00 2017

"""
# import your module here
#import myrca
#rca = myrca.RCA_Supervised(num_chunks=10, chunk_size=2)
#inner_cov=[]

#import mynca
#nca = mynca.NCA(max_iter=100, learning_rate=0.0001)
#LR=0

import collections
import random
from six.moves import xrange
import numpy as np
import time
from threading import Thread
import functools

# class definition here
class RCA_Supervised():
  def __init__(self, dim=None, num_chunks=100, chunk_size=2): 
    """Initialize the learner.

    Parameters
    ----------
    dim : int, optional
        embedding dimension (default: original dimension of data)
    num_chunks: int, optional
    chunk_size: int, optional
    """
    self.params = {
      'dim': dim,
      'num_chunks':num_chunks,
      'chunk_size':chunk_size
    }
  def _process_inputs(self, X, Y):
#    X = np.asanyarray(X)
    self.X = X
    n, d = X.shape

    if self.params['dim'] is None:
      self.params['dim'] = d
    elif not 0 < self.params['dim'] <= d:
      raise ValueError('Invalid embedding dimension, must be in [1,%d]' % d)

#    Y = np.asanyarray(Y)
    num_chunks = int(Y.max() + 1)
#    print(Y,num_chunks)
    return X, Y, num_chunks, d
  def fit(self, X, labels):
    """Learn the RCA model.

    Parameters
    ----------
    X : (n x d) data matrix
        each row corresponds to a single instance
    chunks : (n,) array of ints
        when ``chunks[i] == -1``, point i doesn't belong to any chunklet,
        when ``chunks[i] == j``, point i belongs to chunklet j.
    """
    data, chunks, num_chunks, d = self._process_inputs(X, labels)
#    print(num_chunks)
    # mean center
    data -= data.mean(axis=0)

    # mean center each chunklet separately
    chunk_mask = chunks != -1
    chunk_data = data[chunk_mask]
    chunk_labels = chunks[chunk_mask]
    for c in xrange(num_chunks):
      mask = chunk_labels == c
      chunk_data[mask] -= chunk_data[mask].mean(axis=0)

    # "inner" covariance of chunk deviations
    inner_cov = np.cov(chunk_data, rowvar=0, bias=1)
    return inner_cov.T



class NCA():
  def __init__(self, num_dims=None, max_iter=200, learning_rate=0.001):
    self.num_dims = num_dims
    self.max_iter = max_iter
    self.learning_rate = learning_rate

  def fit(self, X, labels):
    """
    X: data matrix, (n x d)
    y: scalar labels, (n)
    """
    n, d = X.shape
    num_dims = self.num_dims
    if num_dims is None:
        num_dims = d
    # Initialize A to a scaling matrix
    A = np.zeros((num_dims, d))
    np.fill_diagonal(A, 1./(np.maximum(X.max(axis=0)-X.min(axis=0), EPS)))

    # Run NCA
    dX = X[:,None] - X[None]  # shape (n, n, d)
    tmp = np.einsum('...i,...j->...ij', dX, dX)  # shape (n, n, d, d)x(ij)x(ij)T
    masks = labels[:,None] == labels[None]
    for it in xrange(self.max_iter):
      for i, label in enumerate(labels):
        mask = masks[i]
        Ax = A.dot(X.T).T  # shape (n, num_dims)

        softmax = np.exp(-((Ax[i] - Ax)**2).sum(axis=1))  # shape (n) p(i)
        softmax[i] = 0
        softmax /= softmax.sum()

        t = softmax[:, None, None] * tmp[i]  # shape (n, d, d) p(ij)x(ij)x(ij)T
#        d = softmax[mask].sum() * t.sum(axis=0) - t[mask].sum(axis=0)
        d =t.sum(axis=0) - t[mask].sum(axis=0)/(softmax[mask].sum())
        A += 2*self.learning_rate * A.dot(d)

    self.X_ = X
    self.A_ = A
    self.n_iter_ = it
    return self
# (global) variable definition here
TRAINING_TIME_LIMIT = 60*10
EPS = np.finfo(float).eps
#rca
#rca = RCA_Supervised(num_chunks=10, chunk_size=2)
#inner_cov=[]
#nca
nca =NCA(max_iter=100, learning_rate=0.0001)
#inner_cov=[]
# function definition here
def timeout(timeout):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (func.__name__, timeout))]
            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e
            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except Exception as je:
                print('error starting thread')
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret
        return wrapper
    return deco

@timeout(TRAINING_TIME_LIMIT)
def train(traindata):
    train_nca(traindata)#NCA
#    train_rca(traindata)#RCA
#    time.sleep(1) # 这行仅用于测试训练超时，运行时请删除这行，否则你的TRAINING_TIME_LIMIT将-1s。
    return 0
def generator(X,Y,list_index):
    data_1_x=X.copy()
    data_1_y=Y.copy()
    vali_arr_x=np.take(data_1_x, list_index, axis=0) 
    vali_arr_y=np.take(data_1_y, list_index, axis=0) 
    train_arr_x=np.delete(data_1_x, list_index, axis=0)  
    train_arr_y=np.delete(data_1_y,list_index, axis=0)
    return train_arr_x,train_arr_y,vali_arr_x,vali_arr_y
def KNN_error(train_X,train_Y,test_X,test_Y):
    predict_label = np.zeros(test_X.shape[0])
    K=3
    for i in range(test_X.shape[0]):
        distance_vector = np.zeros(train_X.shape[0])
        for j in range(train_X.shape[0]):
            distance_vector[j] = distance(test_X[i], train_X[j])
        labels_of_K_neighbor = train_Y[distance_vector.argsort()[0:K]]
        predict_label[i] = collections.Counter(labels_of_K_neighbor).most_common(n=1)[0][0]
    test_error=np.sum(predict_label != test_Y)/test_Y.shape[0] 
    return np.mean(test_error)
def crossvalidation(fold,X,Y):
    num=Y.shape[0]
    val_num=int(num/fold)
    list_index= random.sample([i for i in range (num)], num)
    erro=0
    for i in range(fold):
        train_arr_x,train_arr_y,vali_arr_x,vali_arr_y=generator(X,Y,list_index[0+i*val_num:val_num+i*val_num])
        nca.fit(train_arr_x,train_arr_y)
        erro=erro+KNN_error(train_arr_x,train_arr_y,vali_arr_x,vali_arr_y)
#        print(KNN_error(train_arr_x,train_arr_y,vali_arr_x,vali_arr_y))
    return erro
def tune_nca(X,Y):
    fold=10
#        TRAINING_TIME_LIMIT = 60*10*10 
    leara=[0.1,0.01,0.001,0.0001,0.00001]
    min_erro=1*fold
    for i in range(len(leara)):
        nca.learning_rate=leara[i]
        erro=crossvalidation(fold,X,Y)
        print (erro)
        if erro<min_erro:
            min_erro=erro
            LR=leara[i]
    print("LR:",LR)
    return LR
def train_nca(traindata): # NCA
    X = traindata[0].copy()
    Y = traindata[1].copy()
#    nca.learning_rate=tune_nca(X,Y) #调节学习率时调用
    nca.fit(X, Y)
#    
def train_rca(traindata): # RCA
    X = traindata[0].copy()
    Y = traindata[1].copy()
    global inner_cov
    inner_cov=rca.fit(X, Y)
def Euclidean_distance(inst_a, inst_b):
    return np.linalg.norm(inst_a - inst_b)
def distance(inst_a, inst_b):
    dist =np.sqrt((inst_a-inst_b).dot(nca.A_.T).dot(nca.A_).dot((inst_a-inst_b).T)) #NCA
#    dist =np.sqrt((inst_a-inst_b).dot(np.linalg.inv(inner_cov)).dot((inst_a-inst_b).T)) #RCA 
    return dist

# main program here
if  __name__ == '__main__':
   
    pass