import numpy as np
import trainLinearReg as tlr
import linearRegCostFunction as lrcf


def learning_curve(X, y, Xval, yval, lmd):
    # Number of training examples
    m = X.shape[0]

    # You need to return these values correctly
    error_train = np.zeros(m)
    error_val = np.zeros(m)


    # ===================== Your Code Here =====================
    # Instructions : Fill in this function to return training errors in
    #                error_train and the cross validation errors in error_val.
    #                i.e., error_train[i] and error_val[i] should give you
    #                the errors obtained after training on i examples
    #
    # Note : You should evaluate the training error on the first i training
    #        examples (i.e. X[:i] and y[:i])
    #
    #        For the cross-validation error, you should instead evaluate on
    #        the _entire_ cross validation set (Xval and yval).
    #
    # Note : If you're using your cost function (linear_reg_cost_function)
    #        to compute the training and cross validation error, you should
    #        call the function with the lamdba argument set to 0.
    #        Do note that you will still need to use lamdba when running the
    #        training to obtain the theta parameters.
    #
    # ==========================================================
    for i in range(1,m+1):
        theta = tlr.train_linear_reg(X[:i,:], y[:i], lmd)
        """
        error_train[i-1] = np.sum((np.matmul(X[:i,:],theta) - y[:i]) ** 2)/(2*i)
        error_val[i-1] = np.sum((np.matmul(Xval,theta) - yval) ** 2)/(2*Xval.shape[0])
        """
        error_train[i-1],_ = lrcf.linear_reg_cost_function(theta, X[:i,:], y[:i], lmd=0)
        error_val[i-1],_ = lrcf.linear_reg_cost_function(theta, Xval, yval, lmd=0)
        
    return error_train, error_val



