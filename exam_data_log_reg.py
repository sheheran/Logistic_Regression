#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
#train test split
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# #mport data using pands
# Data = pd.read_csv('breast-cancer-wisconsin.data.txt', sep=",", header=None)
# Data.columns = ['id','clump_thickness','unif_cell_size','unif_cell_shape','marg_adhesion','single_epith_cell_size','bare_nuclei','bland_chrom','norm_nuclioli','mitoses','class']
# # #Data.replace('?', 0, inplace=True)
# Data.drop(['id'], 1, inplace=True)


 
data = 'exam_data.csv'
Names = ['mark1','mark2','class']
Data = pd.read_csv(data, names=Names)

data = [Data]

mapping_cls = {'pass':1,'fail':0}
for instance in data:
	instance['class'] = instance['class'].map(mapping_cls)
#print(Data.head())

#taking features and labels seperatly
train_data = Data.drop('class', axis=1)
target = Data['class']

#Scatter plot
#labels
y = target
#pass
passed = Data.loc[y == 1]
#loc is used The loc() function is used to access a group of rows and columns by label(s).
#fail
failed = Data.loc[y == 0]

#iloc is used when the index label of a data frame, in case the user doesn't know the index label. Rows can be extracted.
plt.scatter(passed.iloc[:,0], passed.iloc[:,1], s=10, label='passed')
plt.scatter(failed.iloc[:,0], failed.iloc[:,1], s=10, label='failed')
plt.xlabel('exam one')
plt.ylabel('exam two')
plt.title('Data Visualization')
plt.legend()
plt.show()

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
theta = np.random.rand(3,1)
#print(cal_cost(theta,X_train,y_train))

#gradient descent
def gradient_descent(x, y, theta, iteration=100, learingn_rate=0.01):
	#x - feature matrix
	#y - labels
	#iterations - running times
	m = len(y)
	cost_history = np.zeros(iteration)
	theta_history = np.zeros((iteration, 3))
	for i in range(iteration):

		#predict values
		predict = sigmoid(np.dot(x,theta))
		theta = theta - (1/m)*learingn_rate*(x.T.dot((predict - y)))
		theta_history[i,:] = theta.T
		cost_history[i] = cal_cost(theta,x,y)

	return theta, cost_history, theta_history

tr = 0.0001
n_itr = 30
theta, cost_history,theta_history = gradient_descent(X_train,Y_train,theta,n_itr)
# print(sigmoid(np.dot(X_train,theta)))
# print(cost_history)

Y = cost_history
X = list(range(1,(n_itr+1)))

figsize = (10, 8)
plt.plot(X, Y, 'o', ls='-')
plt.show()

