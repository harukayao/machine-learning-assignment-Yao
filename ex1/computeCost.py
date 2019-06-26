import numpy as np


def compute_cost(X, y, theta):
    # Initialize some useful values
    m = y.size
    cost = np.sum((np.matmul(X, theta) - y) ** 2) / (2 * m)

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta.
    #                You should set the variable "cost" to the correct value.


    # ==========================================================

    return cost
