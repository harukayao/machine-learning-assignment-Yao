import numpy as np


def select_threshold(yval, pval):
    f1 = 0

    # You have to return these values correctly
    best_eps = 0
    best_f1 = 0

    for epsilon in np.linspace(np.min(pval), np.max(pval), num=1001):
        # ===================== Your Code Here =====================
        # Instructions: Compute the F1 score of choosing epsilon as the
        #               threshold and place the value in F1. The code at the
        #               end of the loop will compare the F1 score for this
        #               choice of epsilon and set it to be the best epsilon if
        #               it is better than the current choice of epsilon.
        #
        # Note : You can use predictions = pval < epsilon to get a binary vector
        #        of False(0)'s and True(1)'s of the outlier predictions
        #
        # ==========================================================
        predictions = (pval < epsilon).reshape((-1,1))
        X = np.c_[predictions, yval]
        tp = np.sum((X[:,0] == 1) & (X[:,1] == 1))
        fp = np.sum((X[:,0] == 1) & (X[:,1] == 0))
        fn = np.sum((X[:,0] == 0) & (X[:,1] == 1))
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = (2 * prec * rec) / (prec + rec)
        
        if f1 > best_f1:
            best_f1 = f1
            best_eps = epsilon

    return best_eps, best_f1


