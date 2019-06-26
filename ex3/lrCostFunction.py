import numpy as np
from sigmoid import *

 
def lr_cost_function(theta, X, y, lmd):
    m = y.size

    # You need to return the following values correctly
    cost = 0
    grad = np.zeros(theta.shape)

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta
    #                You should set cost and grad correctly.
    #
    # =========================================================
    
    hypo_func = sigmoid(np.matmul(X,theta))
    #cost = ((-np.dot(y, np.log(hypo_func))) - np.dot((1-y), np.log(1 - hypo_func)))/m
    cost = ((-np.dot(y, np.log(hypo_func))) - np.dot((1-y), np.log(1 - hypo_func)))/m + np.sum(theta[1:]**2) * lmd / (2 * m)
    diff = hypo_func - y 
    #grad = np.matmul(X.T, diff) / m
    theta[0] = 0
    grad = np.matmul(X.T, diff) / m + lmd * theta / m
            
    return cost, grad

