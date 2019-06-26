import numpy as np
from sigmoid import *


def cost_function(theta, X, y):
    m = y.size

    # You need to return the following values correctly
    #cost = 0
    grad = np.zeros(theta.shape)

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta
    #                You should set cost and grad correctly.
    #


    # ===========================================================
    hypo_func = sigmoid(np.matmul(X,theta))
    cost = ((-np.dot(y, np.log(hypo_func))) - np.dot((1-y), np.log(1 - hypo_func)))/m
    diff = hypo_func - y
    for j in range(len(grad)):
            grad[j] = np.sum(np.multiply(diff, X[:,j])) / m
    
    return cost, grad
