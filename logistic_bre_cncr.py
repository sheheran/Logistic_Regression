#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression 
#train test split
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#import data using pands
df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', 0, inplace=True)
df.drop(['id'], 1, inplace=True)

#add a column of 1s to train set
X = np.array(df.drop(['class'],1)) #features
y = np.array(df['class']) #labels

# a = 321
# for i  in df:
# 	plt.subplot(a)
# 	plt.scatter(i, y, s=80, c='g', marker='+')
# 	plt.show()
# 	a +=1
# #creating scatter plots


#take collumas a list put variables in column to df['here']

m,n = X.shape
ones = np.ones((m,1))
X = np.hstack((ones, X))
#shuffel and divide data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = LogisticRegression(solver='liblinear',multi_class='ovr')
clf.fit(X_train,y_train)
accuracy = clf.score(X_test,y_test)
print(accuracy)

