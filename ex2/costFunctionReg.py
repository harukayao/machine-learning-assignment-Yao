import numpy as np
from sigmoid import *


def cost_function_reg(theta, X, y, lmd):
    m = y.size

    # You need to return the following values correctly
    cost = 0
    grad = np.zeros(theta.shape)

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta
    #                You should set cost and grad correctly.
    #


    # ===========================================================
    hypo_func = sigmoid(np.matmul(X,theta))
    cost = ((-np.dot(y, np.log(hypo_func))) - np.dot((1-y), np.log(1 - hypo_func)))/m + np.sum(theta[1:]**2) * lmd / (2 * m)
    diff = hypo_func - y
    grad[0] = np.sum(np.multiply(diff, X[:,0])) / m
    for j in range(1,len(grad)):
            grad[j] = np.sum(np.multiply(diff, X[:,j])) / m + lmd * theta[j] / m
            
    
            
    return cost, grad
