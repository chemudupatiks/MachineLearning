# load needed libraries 
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

# load data 
data = np.loadtxt('stu.csv', delimiter=',')
sample = data[:,0:-1] # here, sample matrix = feature matrix
label = data[:,-1]
[n,p] = sample.shape
# split data into training and testing sets 
# this time, we will manual split data (75% for training, 25% for testing)
sample_train = sample[0:int(0.75*n),:]
label_train = label[0:int(0.75*n)]
sample_test = sample[int(0.75*n):,:]
label_test = label[int(0.75*n):]

# data preprocessing (optional) -- this time, let's use standarization
# you can remove this step by commenting out the corresponding lines 
# (figure out the syntax yourself)
scaler = preprocessing.StandardScaler().fit(sample_train) # build scalar using training sample 
# print("sample_train before transform: \n", sample_train)
sample_train = scaler.transform(sample_train) # rescale training sample  
# print("sample_train after transform: \n", sample_train)
sample_test = scaler.transform(sample_test) # rescale testing sample but using the scalar learned from training sample 

# model construction -- we will construct a ridge regression model 
# model selection -- let's try {1e0,1e1,1e2,1e3,1e4,1e5}
# (figure out the syntax yourself)
model = Ridge(alpha = 1e5)

# model training on training sample and its label 
# (figure out the syntax yourself)
model.fit(sample_train, label_train)

# model evaluation 
# -- training error -- 
# apply model to predict labels of training sample 
# (figure out the syntax yourself)
label_train_predicted = model.predict(sample_train)
# print("label_train_predicted: ", label_train_predicted)
# evaluate MSE of predictions 
# (figure out the syntax yourself)
mse_train = mean_squared_error(label_train, label_train_predicted)
# -- testing error -- 
# apply model to predict labels of testing sample 
label_test_predicted = model.predict(sample_test)
# evaluate MSE of predictions 
mse_test = mean_squared_error(label_test, label_test_predicted)

# we're done!
# now plot training error and testing error 
print('Training Error: %.4f' % mse_train)
print('Testing Error: %.4f' % mse_test)




