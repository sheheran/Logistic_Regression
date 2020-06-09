#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
#train test split
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


#mport data using pands
Data = pd.read_csv('breast-cancer-wisconsin.data.txt', sep=",", header=None)
Data.columns = ['id','clump_thickness','unif_cell_size','unif_cell_shape','marg_adhesion','single_epith_cell_size','bare_nuclei','bland_chrom','norm_nuclioli','mitoses','class']
# #Data.replace('?', 0, inplace=True)
Data.drop(['id'], 1, inplace=True)

#taking features and labels seperatly
train_data = Data.drop('class', axis=1)
target = Data['class']


# #importing data from text file using pands
# Data = pd.read_csv('insurance.txt', sep=",", header=None)
# Data.columns = ["age", "bought_insurance"]
# #print(data.head())

# #taking features and labels seperatly
# train_data = Data.drop('age', axis=1)
# target = Data['bought_insurance']



# a = 321
# plt.subplot(a)
# plt.scatter(Data['clump_thickness'], Data['class'], s=80, c='g', marker='o')
# plt.show() 

#add a coulm of 1s to train test
m,n = train_data.shape
ones = np.ones((m,1))
train_data = np.hstack((ones, train_data))

#split train test data set
X_train, X_test, y_train, Y_test = train_test_split(train_data,target, test_size=0.2)

Y_train = np.array([y_train], dtype=np.float64)
Y_train = Y_train.T

#sigmoid function
#z = 1 / (1 + np.exp(-#function))
def sigmoid(x):
	return 1 / (1 + np.exp(-x))


#cost function
def cal_cost(theta,train_set,target):
	m = len(target)
	pred = sigmoid(np.dot(train_set,theta))
	#cost function
	cost = (-1/m)*np.sum(((target )*(np.log(pred))) + ((1 - target)*(np.log(1 - pred))))
	return cost

#finding the hypothesis of the function
theta = np.random.rand(10,1)
#print(cal_cost(theta,X_train,y_train))

#gradient descent
def gradient_descent(x, y, theta, iteration=100, learingn_rate=0.01):
	#x - feature matrix
	#y - labels
	#iterations - running times
	m = len(y)
	cost_history = np.zeros(iteration)
	theta_history = np.zeros((iteration, 10))
	for i in range(iteration):

		#predict values
		predict = sigmoid(np.dot(x,theta))
		theta = theta - (1/m)*learingn_rate*(x.T.dot((predict - y)))
		theta_history[i,:] = theta.T
		cost_history[i] = cal_cost(theta,x,y)

	return theta, cost_history, theta_history

tr = 0.01
n_itr = 100
theta, cost_history,theta_history = gradient_descent(X_train,Y_train,theta,n_itr)
#print(sigmoid(np.dot(X_train,theta)))
print(cost_history)

X = cost_history
Y = list(range(1,(n_itr+1)))
figsize = (10, 8)
plt.plot(X, Y, 'o', ls='-')
plt.show()