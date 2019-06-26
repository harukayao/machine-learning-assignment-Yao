import numpy as np
from sigmoid import *


def threshold(s):
    if s >= 0.5:
        return 1
    else:
        return 0
    
    
def predict(theta, X):
    #m = X.shape[0]

    # Return the following variable correctly
    #p = np.zeros(m)

    # ===================== Your Code Here =====================
    # Instructions : Complete the following code to make predictions using
    #                your learned logistic regression parameters.
    #                You should set p to a 1D-array of 0's and 1's
    #


    # ===========================================================
    p = sigmoid(np.matmul(X,theta))
    p = np.vectorize(threshold)(p)
    return p
