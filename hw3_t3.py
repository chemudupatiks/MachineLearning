import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import random


data = np.genfromtxt('crimerate.csv', delimiter=',')
sample = data[:,0:-1]
label = data[:,-1]
[n,p] = sample.shape

sample_train = sample[0:int(0.75*n),:]
label_train = label[0:int(0.75*n)]
sample_test = sample[int(0.75*n):,:]
label_test = label[int(0.75*n):]

# here, augment raw data with a constant feature of 1 (to include the bias term)
# you may use function "np.c_" to concatenate matrices and "np.ones" to get vector of 1 
sample_train = np.c_[np.ones(sample_train.shape[0]), sample_train]
sample_test =  np.c_[np.ones(sample_test.shape[0]), sample_test]

# now, implement Lasso 
# you can first randomly initialize everything here
beta = []
for i in range(sample_train.shape[1]):
    if i == 0:
        beta.append(0)
    else:
        beta.append(random.uniform(-1,1))

# try lambda = 1e2 and 1e3, remember to set it to zero when updating beta_{0}
lamda = 1e2
num_iter = 1e3
counter = 0
mse_train = []
mse_test = []
index_iter = []
num_selectedfeature= []


def get_sigma_two_x_a(j):
    result = 0
    beta_copy = beta.copy()
    beta_copy[j] = 0
    Aj = np.matmul(sample_train, beta_copy) - label_train
    for i in range(sample_train.shape[0]):
        result =  result + (sample_train[i][j] * Aj[i])
    return 2*result

def get_sigma_xij_square(j):
    result = 0
    for i in range(sample_train.shape[0]):
        result = result + sample_train[i][j]*sample_train[i][j]
    return 2* result


while(counter <= num_iter):
    
    j =  np.random.randint(0, len(beta))
    
    if j == 0:
        # print("if  : ", str(counter))
        beta_j = 0
        for i in range(sample_train.shape[0]):
            beta_j = beta_j + (np.matmul(sample_train[i][1:].transpose(), beta[1:]) - label_train[i])
        beta[j] = (-1/sample_train.shape[0])*beta_j
    else:
        # print("else: ", str(counter))
        sigma_xa = get_sigma_two_x_a(j)
        sigma_xij_square = get_sigma_xij_square(j)
        if sigma_xa < -1*lamda:
            beta[j] = (-lamda - sigma_xa)/sigma_xij_square
        elif sigma_xa > lamda:
            beta[j] = (lamda - sigma_xa)/sigma_xij_square
        else:
            beta[j] = 0            

    # at the end of every iteration, save your training & testing errors 
    # you will need to initialize both variables before the loop, though 
    label_train_predicted = np.matmul(sample_train, beta)
    label_test_predicted = np.matmul(sample_test, beta)
    mse_train.append(mse(label_train, label_train_predicted))
    mse_test.append(mse(label_test, label_test_predicted))
    num_selectedfeature.append(np.count_nonzero(beta))
    # this simply saves all iteration indices for plotting 
    index_iter.append(counter) 
    counter += 1

print("Converged Testing MSE = ",mse_test[-1])
       
fig = plt.figure()  
plt.plot(index_iter, mse_train, label='training mse')
plt.plot(index_iter, mse_test, label='testing mse')
plt.legend()
plt.show()   

fig = plt.figure()  
plt.plot(index_iter, num_selectedfeature, label='number of selected features')
plt.legend()
plt.show()

# print("Converged Testing MSE = ",mse_test[-1])
# print("done")