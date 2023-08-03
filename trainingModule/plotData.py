from turtle import width
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import _tkinter

from trainingModule.visualizemodel import plot_decision_regions

def plotPCA(X_train, X_test, y_train, y_test, classifier):
    pca = PCA(n_components=2)
    X_com = np.vstack((X_train, X_test))
    Y_com = np.hstack((y_train, y_test))
    Xt = pca.fit_transform(X_com)

    test_idx = range(X_train.shape[0], X_com.shape[0])

    plot_decision_regions(Xt, Y_com, classifier, test_idx=test_idx, pca=pca)

def plotBar(x,y, tick_label,xlabel="x", ylabel="y"):
    fig, ax = plt.subplots()
    ax.bar(x, y, width=0.5)
    ax.set_xlabel(xlabel)
    ax.set_xticklabels(tick_label)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.94, 1.0)
    plt.show()
