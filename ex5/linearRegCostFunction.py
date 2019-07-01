import numpy as np


def linear_reg_cost_function(theta, x, y, lmd):
    # Initialize some useful values
    m = y.size

    # You need to return the following variables correctly
    cost = 0
    grad = np.zeros(theta.shape)

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost and gradient of regularized linear
    #                regression for a particular choice of theta
    #
    #                You should set 'cost' to the cost and 'grad'
    #                to the gradient
    #
    # ==========================================================
    hypo_func = np.matmul(x,theta)
    cost = np.sum((hypo_func - y) ** 2) / (2 * m) + np.sum(theta[1:]**2) * lmd / (2 * m)
    diff = hypo_func - y 
    theta_ = theta.copy()
    theta_[0] = 0
    grad = np.matmul(x.T, diff) / m + lmd * theta_ / m

    return cost, grad
