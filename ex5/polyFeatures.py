import numpy as np


def poly_features(X, p):
    # You need to return the following variable correctly.
    X_poly = np.zeros((X.size, p))

    # ===================== Your Code Here =====================
    # Instructions : Given a vector X, return a matrix X_poly where the p-th
    #                column of X contains the values of X to the p-th power.
    #
    # ==========================================================
    X = X.reshape(X.size,1)
    X_poly[:,0] = X[:,0]
    for i in range(1,p):
        X_poly[:,i] = np.power(X_poly[:,0],i+1)
    return X_poly