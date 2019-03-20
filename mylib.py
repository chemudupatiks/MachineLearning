#
# Here, define two functions for your regularized logistic regression model 
# For more examples, see 
# https://docs.python.org/3/tutorial/modules.html
# 

import numpy as np
import matplotlib.pyplot as plt
import random
from math import exp
from math import log
from numpy.linalg import inv

def getPosteriorWhenYZero(X, beta):
    result = np.exp(np.matmul(X.transpose(), beta))
    result = result/(1+result)
    # print(result)
    return result

def getPosteriorWhenYOne(X, beta):
    result = np.exp(np.matmul(X.transpose(), beta))
    result = 1/(1+result)
    # print(res ult)
    return result

def getError(beta, sample, label):
    result = 0
    y_predicted = []
    for i in range(sample.shape[0]):
        x = getPosteriorWhenYOne(sample[i], beta)
        if x>0.5: 
            y_predicted.append(1)
        else:
            y_predicted.append(0)

    no_of_error = 0
    for i in range(len(y_predicted)):
        if y_predicted[i] != label[i]:
            no_of_error += 1
    # print(result)
    result = no_of_error/len(y_predicted)
    return result

# Function 1: define the process of model training as logit_fit
def logit_fit(sample,label,lamda,max_iter):
# input:
#   sample: training data 
#   label: training label
#   lamda: regularization coefficient 
#   max_iter: maximum number of update iteration 
# output: 
#   beta: prediction model 

    beta = np.zeros(sample.shape[1])
    for i in range(sample.shape[1]):
        beta[i] = np.random.normal(0,0.1)

    lamda_matrix = np.identity(sample.shape[1])
    for i in range(sample.shape[1]):
        lamda_matrix[i][i] = lamda
# now implement the iterative update of regularized logistic regression
    # err_train = []
    W = np.identity(sample.shape[0])
    p = np.zeros(sample.shape[0])

    for t in range(max_iter):
        for i in range(sample.shape[0]):
            W[i][i] = getPosteriorWhenYOne(sample[i][:],beta) * getPosteriorWhenYZero(sample[i][:], beta)
        # print(W)
        
        for i in range(sample.shape[0]):
            # print(i)
            p[i] = getPosteriorWhenYOne(sample[i][:], beta)
        # print(p)

        # err_train.append(getError(beta, sample, label))

        GPrime = np.matmul(sample.transpose(), (label - p)) + np.multiply(lamda, beta)
        # print(GPrime)
        m = np.matmul(sample.transpose(), (np.matmul(W, sample)))
        # print(m.shape)
        GDoublePrime = np.add(m ,lamda_matrix)
        # print(GDoublePrime)
        l = np.matmul(inv(GDoublePrime), GPrime) 
        # print(l)
        beta = beta - l
    # finally, get the model parameter 'beta'
    return beta


    
# Function 2: define the process of predction as logit_pred
def logit_pred(sample,beta):
# input: 
#   sample: testing data 
#   beta: model 
# output: 
#   label_pred: predicted labels of sample 
    label_predicted = []
    for i in range(sample.shape[0]):
        x = getPosteriorWhenYOne(sample[i], beta)
        if x>0.5: 
            label_predicted.append(1)
        else:
            label_predicted.append(0)
 
    return label_predicted

    


