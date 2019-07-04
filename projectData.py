import numpy as np


def project_data(X, U, K):
    # You need to return the following variables correctly.
    Z = np.zeros((X.shape[0], K))

    # ===================== Your Code Here =====================
    # Instructions: Compute the projection of the data using only the top K
    #               eigenvectors in U (first K columns).
    #               For the i-th example X[i], the projection on to the k-th
    #               eigenvector is given as follows:
    #                   x = X(i, :)';
    #                   projection_k = x' * U(:, k);
    #                   (above is octave code)
    #
    # ==========================================================
    U_reduce = U[:, 0:K]
    Z = np.zeros((len(X), K))
    for i in range(len(X)):
        x = X[i,:]
        projection_k = np.dot(x, U_reduce)
        Z[i] = projection_k
    """
    for i in range(X.shape[0]):
        for j in range(K):
            Z[i][j] = np.matmul(X[i,:].reshape(1,X[i,:].size),U[:,j])
    """
    return Z

