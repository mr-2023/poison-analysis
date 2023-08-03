import numpy as np
from sklearn.model_selection import train_test_split

def splitData(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1, stratify=y)
    #[0の個数 1の個数]
    print('Labels counts in y:', np.bincount(y))
    print('Labels counts in y_train:', np.bincount(y_train))
    print('Labels counts in y_test:', np.bincount(y_test))
    return X_train, X_test, y_train, y_test

def checkTrue(y1, y2):
    for i,j in zip(y1,y2):
        if i!=j:
            print("y1 and y2 mismatch")
    print("y1 and y2 matching succeed!")