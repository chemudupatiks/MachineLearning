import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn import preprocessing

data = np.loadtxt('stu.csv', delimiter=',')
sample = data[:,0:-1] 
label = data[:,-1]
[n,p] = sample.shape

sample_train = sample[0:int(0.75*n),:]
label_train = label[0:int(0.75*n)]
sample_test = sample[int(0.75*n):,:]
label_test = label[int(0.75*n):]

# choose from the following set of candidate alphas 
alpha_set = {1e1,1e3,1e5}

# implement K-Fold cross validation from scractch, set K = 5
K = 5 
avg_errors_folds = []
# try each alpha 
for alpha_candidate in alpha_set:
    
    # construct your model with candidate alpha 
    model = Ridge(alpha = alpha_candidate)
    [x, y] = sample_train.shape
    errors_folds = []
    for i in range(K):
    	sample_test_part = sample_train[int((i)*x/K):int((i+1)*x/K), :]
    	label_test_part = label_train[int((i)*x/K):int((i+1)*x/K)]
    	sample_train_part1 = sample_train[0:int((i)*x/K), :]
    	sample_train_part2 = sample_train[int((i+1)*x/K):, :]
    	label_train_part1 = label_train[0:int((i)*x/K)]
    	label_train_part2 = label_train[int((i+1)*x/K):]
    	sample_train_part = np.concatenate((sample_train_part1, sample_train_part2))
    	label_train_part = np.concatenate((label_train_part1, label_train_part2))
    	model.fit(sample_train_part,label_train_part)
    	label_test_predicted = model.predict(sample_test_part)
    	mse_test = mean_squared_error(label_test_part,label_test_predicted)
    	print(mse_test)
    	errors_folds.append(mse_test)
    print("------------------------------------")
    avg_errors_folds.append(np.mean(errors_folds))
    
print(avg_errors_folds)
# now set the optimal alpha selected by cross-validation 
alpha_opt = min(avg_errors_folds)
# print (alpha_opt)
# after finding the optimal hyper-parameter alpha_opt, 
# retrain ridge regression on the training sample with alpha_opt 
# and evaluat the model on testing set 

model = Ridge(alpha = alpha_opt)

model.fit(sample_train,label_train)

label_test_predicted = model.predict(sample_test)
mse_test = mean_squared_error(label_test,label_test_predicted)

print('Testing Error: %.4f' % mse_test)




