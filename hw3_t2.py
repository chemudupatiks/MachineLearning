
# load needed libraries 
import numpy as np

data = np.loadtxt('crimerate.csv', delimiter=',')

sample = data[:,0:-1] 
label = data[:,-1] 
[n,p] = sample.shape

sample_train = sample[0:int(0.75*n),:]
# print(sample_train.shape[0])
label_train = label[0:int(0.75*n)]
sample_test = sample[int(0.75*n):,:]
label_test = label[int(0.75*n):]

# get the index ste of minority and majority communities in training & testing sets 
# you may use function "np.where" and add [0] at the end to convert results into array 
# recall that the 3rd column is percentage of African American residents 
# index_minority_train = sample_train[np.where(sample_train[:, 2]>=0.5)[0], :]
index_minority_train = np.where(sample_train[:, 2]>=0.5)[0]
# print(index_minority_train)
# print(sample_train[np.where(sample_train[:, 2]>=0.5)[0], :].shape)
# index_majority_train = sample_train[np.where(sample_train[:, 2]<0.5)[0], :]
index_majority_train = np.where(sample_train[:, 2]<0.5)[0]
# print(index_majority_train)
# print(np.where(sample_train[:, 2]<0.5)[0].shape)
index_minority_test = np.where(sample_test[:, 2]>=0.5)[0]
# index_minority_test = sample_test[np.where(sample_test[:, 2]>=0.5)[0], :]
# print(np.where(sample_test[:, 2]>=0.5)[0].shape)
index_majority_test = np.where(sample_test[:, 2]<0.5)[0]
# index_majority_test = sample_test[np.where(sample_test[:, 2]<0.5)[0], :]
# print(np.where(sample_test[:, 2]<0.5)[0])
# print(np.where(sample_test[:, 2]<0.5)[0].shape)

# here, augment raw data with a constant feature of 1 (to include the bias term)
# you may use function "np.c_" to concatenate matrices and "np.ones" to get vector of 1 
sample_train = np.c_[np.ones(sample_train.shape[0]), sample_train]
sample_test = np.c_[np.ones(sample_test.shape[0]), sample_test]

# fix lambda to 1e1 (but feel free to try other values after implementation)
lamda = 20
# lambda_matrix = np.identity()

# now implement weighted ridge regression based on your derived analytic solution 
# use "W" to denote your weight matrix/vector/set
weight_minor = 100
W = np.identity(sample_train.shape[0])
for i in index_minority_train:
	# print(i)
	W[i][i] = weight_minor

# for i in range(len(W)):
# 	print(W[i], "\n")

# print(W.shape)
# print(sample_train.transpose().shape)
# print(label_train.shape)
# let's call the final model "beta", which is a (p+1)-dimensional vector 
beta_p1 = np.matmul(np.matmul(sample_train.transpose(), W), sample_train)
# print(beta_p1)
# print(beta_p1.shape) 
lambda_matrix = np.identity(beta_p1.shape[0])
for i in range(beta_p1.shape[0]):
	lambda_matrix[i][i] = lamda

# print(lambda_matrix)
beta_p1 = np.add(beta_p1,lambda_matrix)

beta_p2 = np.matmul(np.matmul(sample_train.transpose(), W), label_train)
# print(beta_p2)
# print(beta_p2.shape) 

inverse = np.linalg.inv(beta_p1)
beta = np.matmul(inverse, beta_p2)
# print(beta)
# print(beta.shape)
# apply beta on training and testing set to get their predictions 
# please implement MSE evaluation from scratch, you cannot use the embedded mse function 
# you may use function "len(x)" to get the number of elements in vector x
label_train_predicted = np.matmul(sample_train, beta)
label_test_predicted = np.matmul(sample_test, beta)

# print(label_train_predicted)
# print(label_test_predicted)

train_diff = (label_train - label_train_predicted)
test_diff = (label_test - label_test_predicted)

for i in range(train_diff.size):
	train_diff[i] = train_diff[i]* train_diff[i]
for i in range(test_diff.size):
	test_diff[i] = test_diff[i]* test_diff[i]

# print(train_diff)
# print(test_diff)
sum_train_diff = 0
sum_test_diff = 0

for i in range(train_diff.size):
	sum_train_diff = sum_train_diff + train_diff[i]
for i in range(test_diff.size):
	sum_test_diff = sum_test_diff + test_diff[i]

# print(sum_train_diff)
# print(sum_test_diff)

mse_train_total = sum_train_diff/len(train_diff)
mse_test_total = sum_test_diff/len(test_diff)

# print(mse_train_total)
# print(mse_test_total)

# now evaluate MSE on minority testing subsample 
# you can use e.g., "predicted_label_test[index_female_test]" to get the prediction on such subsample 
train_diff_minority = train_diff[index_minority_train]
test_diff_minority = test_diff[index_minority_test]

sum_train_diff_minority = 0
sum_test_diff_minority = 0

for i in range(train_diff_minority.size):
	sum_train_diff_minority = sum_train_diff_minority + train_diff_minority[i]
for i in range(test_diff_minority.size):
	sum_test_diff_minority = sum_test_diff_minority + test_diff_minority[i]

mse_train_minority = sum_train_diff_minority/len(train_diff_minority)
mse_test_minority = sum_test_diff_minority/len(test_diff_minority)

# print(mse_train_minority)
# print(mse_test_minority)

#.............................................................

train_diff_majority = train_diff[index_majority_train]
test_diff_majority = test_diff[index_majority_test]

sum_train_diff_majority = 0
sum_test_diff_majority = 0

for i in range(train_diff_majority.size):
	sum_train_diff_majority = sum_train_diff_majority + train_diff_majority[i]
for i in range(test_diff_majority.size):
	sum_test_diff_majority = sum_test_diff_majority + test_diff_majority[i]

mse_train_majority = sum_train_diff_majority/len(train_diff_majority)
mse_test_majority = sum_test_diff_majority/len(test_diff_majority)

# print(mse_train_majority)
# print(mse_test_majority)

#...........................................................
# mse_train_majority = 
# mse_test_majority = 

print('\nTraining Error (Total): %.4f' % mse_train_total)
print('Testing Error (Total): %.4f' % mse_test_total)

print('\nTraining Error (Minority): %.4f' % mse_train_minority)
print('Training Error (Majority): %.4f' % mse_train_majority)

print('\nTesting Error (Minority): %.4f' % mse_test_minority)
print('Testing Error (Majority): %.4f' % mse_test_majority)

# print('\nTesting Error (Minority): %.4f' % mse_test_minority)
# print('\nTesting Error (Majority): %.4f' % mse_test_majority)
# print('\nTesting Error (Total): %.4f' % mse_test_total)





