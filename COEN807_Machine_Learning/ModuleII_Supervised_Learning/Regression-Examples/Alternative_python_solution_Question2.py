#OpTIMAL sOLUTION
import numpy as np
import matplotlib.pyplot as plt

#Iteration 0
#Inputs
X =  [[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
X = np.array(X)
print('Input Examples')
print(X)

X_X_dot = np.dot(np.transpose(X),X)
print(X_X_dot)

X_Xinv = np.linalg.inv(np.dot(np.transpose(X),X))

print(X_Xinv)

#Actual Target
D = [0, 1, 1, 1]

W_1 = np.dot(np.dot(np.transpose(D),X),np.linalg.inv(np.dot(np.transpose(X),X)))

print('Updated Weights')
print(W_1)


Y_1 = np.sum(X*W_1, axis=1)
print('Predicted Outputs')
print(Y_1)

#Error
print('Prediction Error')
E = 0.5*np.sum((D - Y_1)*(D - Y_1), axis=0)
print(E)

