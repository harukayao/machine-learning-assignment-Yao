import numpy as np


def recover_data(Z, U, K):
    # You need to return the following variables correctly.
    X_rec = np.zeros((Z.shape[0], U.shape[0]))

    # ===================== Your Code Here =====================
    # Instructions: Compute the approximation of the data by projecting back
    #               onto the original space using the top K eigenvectors in U.
    #
    #               For the i-th example Z[i], the approximate
    #               recovered data for dimension j is given as follows:
    #                   v = Z(i, :)';
    #                   recovered_j = v' * U(j, 1:K)';
    #                   (above is octave code)
    #
    # ==========================================================
    """
    for i in range(Z.shape[0]):
        for j in range(U.shape[0]):
            X_rec[i][j] = np.matmul(Z[i,:].reshape(Z.shape[1],1),U[j,0:K])
    """
    for i in range(len(Z)):
        v = Z[i,:]
        for j in range(np.size(U,1)):
            recovered_j = np.dot(v.T,U[j,0:K])
            X_rec[i][j] = recovered_j   
    
    return X_rec

