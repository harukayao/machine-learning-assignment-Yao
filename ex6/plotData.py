import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_data(X, y):
    plt.figure()
    df = pd.DataFrame(X)
    df['y'] = y
    pos = df[df['y']==1]
    neg = df[df['y']==0]
    plt.scatter(pos[0],pos[1],marker='+', c='black')
    plt.scatter(neg[0],neg[1],s=10, c='orange')

    # ===================== Your Code Here =====================
    # Instructions : Plot the positive and negative examples on a
    #                2D plot, using the marker="+" for the positive
    #                examples and marker="o" for the negative examples
    

