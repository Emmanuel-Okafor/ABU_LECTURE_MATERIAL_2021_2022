# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 10:41:11 2021

@author: user
"""

import numpy as np
from sklearn import  linear_model

#Training Data
X = np.array([[1],[2],[3],[4]])
print(X)

y = np.array([[0],[1],[1],[2]])
print(y)

#Create linear regression object
reg = linear_model.LinearRegression()

#Train the model using the training sets
reg.fit(X, y)

# The coefficients
print('Coefficients: \n', reg.coef_,  reg.intercept_)

#Testing Data
X_test = np.array([[26] ])

print('Predictions using the testing set')
y_test = reg.predict(X_test)

print(y_test)




