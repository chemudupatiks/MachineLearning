import numpy as np
from numpy.linalg import norm

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
# now split data
sample_train = sample[0:int(0.75*n),:]
label_train = label[0:int(0.75*n)]
sample_test = sample[int(0.75*n):,:]
label_test = label[int(0.75*n):]

k = 1
# now, implement weighted kNN

# tips: 
# - you don't have to write a for loop to go over every testing instance 
#   instead, you may implement kNN using matrix operations 
# - look for functions that can compute pair-wise distance between two sets of instances 
#   one set is testing sample, and the other set is training sample  
# - you first get all pairwise distances between training instances and testing instances 
#   then, do nearest neighbor search and the rest analysis based on that distance matrix 
# - think about how to convert those distances into weights based on (7)
# - think about how to convert weights into votes based on (6)

# print(sample_train.shape)
# print(sample_test.shape)
distance_matrix = np.zeros((sample_test.shape[0], sample_train.shape[0]))
# print(distance_matrix.shape)
for i in range(distance_matrix.shape[0]):
	for j in range(distance_matrix.shape[1]):
		distance_matrix[i][j] = norm(sample_train[j]-sample_test[i])


# print(np.argsort(distance_matrix[1])[:k].shape)

knn = np.zeros((sample_test.shape[0], k))
weight = np.zeros((sample_test.shape[0], k))
for i in range(knn.shape[0]):
	temp_index = np.argsort(distance_matrix[i])[:k]
	weight[i] = distance_matrix[i][temp_index]
	knn[i] = label_train[temp_index]

# print(knn)
# print(knn.shape)
# print(weight)
# print(weight.shape)

pred_label = np.zeros(sample_test.shape[0])
# print(pred_label.shape)

for i in range(knn.shape[0]):
	prob_array= np.zeros(3)
	total = np.sum(weight[i])
	for j in range(3):
		temp_index = np.argwhere(knn[i]==j).flatten()
		# print(temp_index)
		if temp_index.size == 0:
			prob_array[j] = 0
		else:
			prob_array[j] = np.sum(weight[i][temp_index])
	prob_array = prob_array/total
	pred_label[i] =  np.argmax(prob_array)

# print(np.argwhere(pred_label == 2).flatten())

no_of_error = 0
for i in range(pred_label.size):
    if pred_label[i] != label_test[i]:
        no_of_error += 1
# print(result)
result = no_of_error/len(pred_label)

print(result)


