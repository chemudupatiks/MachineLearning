import numpy as np
import matplotlib.pyplot as plt
import random
from math import exp
from math import log
from numpy.linalg import inv

def getPosteriorWhenYZero(X, beta):
	# print(X.transpose().shape, beta.shape)
	# print(np.matmul(X.transpose(), beta))
	# print(X[1:].transpose())
	# print(beta)
	result = np.exp(np.matmul(X.transpose(), beta))
	result = result/(1+result)
	# print(result)
	return result

def getPosteriorWhenYOne(X, beta):
	result = np.exp(np.matmul(X.transpose(), beta))
	result = 1/(1+result)
	# print(res ult)
	return result
# 
# result = 2
# result = 1/result
# print(result)



data = np.genfromtxt('crimerate.csv', delimiter=',')
sample = data[:,0:-1]
label = data[:,-1]
[n,p] = sample.shape

# you may want to preprocess the raw data set 
# first, threshold the continuous crime rates and convert them into 2 or 3 classes
for i in range(len(label)):
	if(label[i] > 0.5):
		label[i] = 1
	else:
		label[i] = 0

# print(label)

# second, don't forget to include a column of constant feature 1 (for including the bias term in the model)
sample = np.c_[np.ones(sample.shape[0]), sample]
# print(sample)

# now, split sample into training and testing sets 
sample_train = sample[0:int(0.75*n),:]
label_train = label[0:int(0.75*n)]
sample_test = sample[int(0.75*n):,:]
label_test = label[int(0.75*n):]

# pick up an optimal lamba in anyway you like 
lamda = 10
lamda_matrix = np.identity(sample_train.shape[1])
for i in range(sample_train.shape[1]):
		lamda_matrix[i][i] = lamda

# decide maximum number of update iterations 
# however we could always stop earlier when another stopping criterion is met  
max_iter = 15

def getTrainingError(beta):
	result = 0
	y_predicted = []
	for i in range(sample_train.shape[0]):
		x = getPosteriorWhenYOne(sample_train[i], beta)
		if x>0.5: 
			y_predicted.append(1)
		else:
			y_predicted.append(0)

	no_of_error = 0
	for i in range(len(y_predicted)):
		if y_predicted[i] != label_train[i]:
			no_of_error += 1
	# print(result)
	result = no_of_error/len(y_predicted)
	return result

# def getTrainingError(beta):
# 	result = 0
# 	for i in range(sample_train.shape[0]):
# 		result = result + (((label_train[i] -1)*(np.log(getPosteriorWhenYZero(sample_train[i], beta)))) - (label_train[i] * np.log(getPosteriorWhenYOne(sample_train[i], beta)))) 
# 	return result

def getTestingError(beta):
	# result = 0
	# for i in range(sample_test.shape[0]):
	# 	xBeta = np.matmul(sample_test[i].transpose(), beta)
	# 	result = result + (((label_test[i] - 1)*(xBeta)) + np.log(1+np.exp(xBeta))) + lamda*np.matmul(beta[1:].transpose(), beta[1:])
	# # print(result)
	# return result
	result = 0
	y_predicted = []
	for i in range(sample_test.shape[0]):
		x = getPosteriorWhenYOne(sample_test[i], beta)
		if x>0.5: 
			y_predicted.append(1)
		else:
			y_predicted.append(0)

	no_of_error = 0
	for i in range(len(y_predicted)):
		if y_predicted[i] != label_test[i]:
			no_of_error += 1
	# print(result)
	result = no_of_error/len(y_predicted)
	return result

# random initialization of beta 
beta = np.zeros(sample_train.shape[1])
for i in range(sample_train.shape[1]):
    # if i == 0:
    #     # beta.append(0)
    #     beta[i] = 0
    # else:
        # beta.append(random.uniform(-1,1)) 
    beta[i] = np.random.normal(0,0.1)

# print(beta)
# print(sample_train[1])
# now implement the iterative update of regularized logistic regression
err_train = []
err_test = []
W = np.identity(sample_train.shape[0])
p = np.zeros(sample_train.shape[0])

for t in range(max_iter):
	for i in range(sample_train.shape[0]):
		W[i][i] = getPosteriorWhenYOne(sample_train[i][:],beta) * getPosteriorWhenYZero(sample_train[i][:], beta)
	# print(W)
	
	for i in range(sample_train.shape[0]):
		# print(i)
		p[i] = getPosteriorWhenYOne(sample_train[i][:], beta)
	# print(p)

	err_train.append(getTrainingError(beta))
	err_test.append(getTestingError(beta))

	GPrime = np.matmul(sample_train.transpose(), (label_train - p)) + np.multiply(lamda, beta)
	# print(GPrime)
	m = np.matmul(sample_train.transpose(), (np.matmul(W, sample_train)))
	# print(m.shape)
	GDoublePrime = np.add(m ,lamda_matrix)
	# print(GDoublePrime)
	l = np.matmul(inv(GDoublePrime), GPrime) 
	# print(l)
	beta = beta - l
	# beta = beta - np.matmul(inv(GDoublePrime), GPrime) 
	

# finally, plot convergence curves    
print("testing error:",err_test[len(err_test)-1]) 
print("training error: ",err_train[len(err_train)-1]) 
plt.plot(err_train, label='training error')
plt.plot(err_test, label='testing error')
plt.legend()
plt.show()   

