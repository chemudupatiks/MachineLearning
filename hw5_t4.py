import numpy as np
import matplotlib.pyplot as plt

# here, import your implemented two functions from mylib 
# see `mylib.py' for more instructions 
from mylib import logit_fit, logit_pred, getError

data = np.genfromtxt('crimerate.csv', delimiter=',')
sample = data[:,0:-1]
label = data[:,-1]
[n,p] = sample.shape

# again, you may want to preprocess the raw data set 
# so that the label becomes discrete in three classes 
for i in range(len(label)):
	if(label[i] > 0.66):
		label[i] = 2
	elif(label[i] > 0.33):
		label[i] = 1
	else:
		label[i] = 0
# also, don't forget to include a column of constant feature 
sample = np.c_[np.ones(n), sample]

# now split data
sample_train = sample[0:int(0.75*n),:]
label_train = label[0:int(0.75*n)]
sample_test = sample[int(0.75*n):,:]
label_test = label[int(0.75*n):]

# number of classes 
num_class = 3

# beta is a matrix storing all models 
# the k_{th} column is the model that separates class k from the rest
beta = np.zeros((p+1,num_class)) 

# label_test_pred is a matrix storing all predictions 
# the k_{th} column is the prediction by model k 
label_test_pred = np.zeros((len(label_test),num_class)) 

# now, start one-versus-all strategy 
# for every class k, build a binary classifier 
for k in range(num_class): 
    
    # here, modify your training sample and save it in a temporary set 
    # ... 
    # ... 
    label_train_temp = []
    for i in range(len(label_train)):
    	if label_train[i] == k:
    		label_train_temp.append(0)
    	else:
    		label_train_temp.append(1)

    # print("label_train_temp",label_train_temp)
    
    # next, fit a regularized logistic regression model \
    beta[:,k] = logit_fit( sample_train, label_train_temp, lamda=10 , max_iter=10 )
    
    # next, make prediction on testing sample
    # print("beta[:][k] shape is: ", beta[:][k].shape)
    label_test_pred[:,k] = logit_pred( sample_test, beta[:,k])


# now, aggregate votes from all models and get the final prediction 
# ... 
# ...   
# print(label_test_pred)
label_test_pred_agg = np.zeros(len(label_test))
for i in range(len(label_test)):
	Votes = np.array([0, 0 ,0])
	for j in range(num_class):
		# print
		if label_test_pred[i][j] == 0:
			Votes[j] += 1
		else:
			Votes[:j] += 1
			Votes[j+1:] += 1
	# print(Votes)
	max_index = np.argwhere(Votes == np.amax(Votes)).flatten()
	if len(max_index)>1:
		label_test_pred_agg[i] = max_index[np.random.randint(0, len(max_index))]
	else:
		label_test_pred_agg[i] = max_index[0] 


# print(label_test_pred_agg)

no_of_error = 0
for i in range(len(label_test_pred_agg)):
    if label_test_pred_agg[i] != label_test[i]:
        no_of_error += 1
# print(result)
result = no_of_error/len(label_test_pred_agg)

print(result)

# measure classification error 
# ... 
# ...

