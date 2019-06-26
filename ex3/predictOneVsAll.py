import numpy as np
from sigmoid import *

def predict_one_vs_all(all_theta, X):
    m = X.shape[0]
    num_labels = all_theta.shape[0]

    # You need to return the following variable correctly;
    p = np.zeros(m)

    # Add ones to the X data matrix
    X = np.c_[np.ones(m), X]

    # ===================== Your Code Here =====================
    # Instructions : Complete the following code to make predictions using
    #                your learned logistic regression parameters (one vs all).
    #                You should set p to a vector of predictions (from 1 to
    #                num_labels)
    #
    # Hint : This code can be done all vectorized using the max function
    #        In particular, the max function can also return the index of the
    #        max element, for more information see 'np.argmax' function.
    #
    y = sigmoid(np.matmul(X,all_theta.T))
    p = np.argmax(y,axis=1) 
    p[p==0] = 10
   

    return p
