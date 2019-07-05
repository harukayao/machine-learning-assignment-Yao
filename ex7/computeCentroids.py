import numpy as np
import pandas as pd


def compute_centroids(X, idx, K):
    # Useful values
    (m, n) = X.shape

    # You need to return the following variable correctly.
    #centroids = np.zeros((K, n))

    # ===================== Your Code Here =====================
    # Instructions: Go over every centroid and compute mean of all points that
    #               belong to it. Concretely, the row vector centroids[i]
    #               should contain the mean of the data points assigned to
    #               centroid i.
    #
    # ==========================================================
    df = pd.DataFrame(X)
    df['idx'] = idx
    g = df.groupby('idx')
    centroids = np.zeros((n, K))
    """
    centroids = g[0,1].mean()
    centroids = np.array(centroids)
    """
    for i in range(n):
        centroids[i] = g[i].mean()
    centroids = centroids.T
    
    return centroids
