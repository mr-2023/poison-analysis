import numpy as np
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

def fitModel(X_train, y_train, random_state= 1234, mode="Perceptron", kernel='rbf', gamma= 0.0001, C=100,loss='log', penalty='l2',n_neighbors=6, solver='adam', hidden_layer_sizes=(100,), activation='relu'):
    tol = 1e-3
    if mode == "Perceptron":
        ppn = Perceptron(max_iter=10000, eta0=0.1, tol=tol, random_state=random_state)
        ppn.fit(X_train , y_train)
        return ppn
    elif mode == "SVC":
        svcClassifier = SVC(kernel=kernel, gamma=gamma, C=C)
        svcClassifier.fit(X_train, y_train)
        return svcClassifier
    elif mode == "SGD":
        clf = SGDClassifier(loss=loss, penalty=penalty, max_iter=100000, fit_intercept=True, random_state=random_state, tol=tol)
        clf.fit(X_train, y_train)
        return clf
    elif mode == "k-neighbors":
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        return knn
    elif mode == "neural network":
        clf = MLPClassifier(solver=solver, random_state=random_state, max_iter=100000,hidden_layer_sizes=hidden_layer_sizes, activation=activation)
        clf.fit(X_train, y_train)
        return clf
    elif mode == "random forest":
        clf = RandomForestClassifier(random_state=random_state)
        clf.fit(X_train,y_train)
        return clf
    return 0

def fitModelFailed(X_train, y_train, y_pred, random_state= 1234, mode="Perceptron", kernel='rbf', gamma= 0.0001, C=100,loss='log', penalty='l2',n_neighbors=6, solver='adam', hidden_layer_sizes=(100,), activation='relu'):
    X_train_f = [X_train[i] for i in range(y_train.shape[0]) if y_pred[i] != y_train[i]]
    y_train_f = [y_train[i] for i in range(y_train.shape[0]) if y_pred[i] != y_train[i]]
    X_train_f[len(X_train_f):len(X_train_f)] = X_train
    y_train_f[len(y_train_f):len(y_train_f)] = y_train
    print("Failed size:X ", len(X_train_f), ",:y ", len(y_train_f))
    return fitModel(X_train_f, y_train_f, random_state= random_state, mode=mode, kernel=kernel, gamma= gamma, C=C,loss=loss, penalty=penalty,n_neighbors=n_neighbors, solver=solver, hidden_layer_sizes=hidden_layer_sizes, activation=activation)
    
#引用:https://dodotechno.com/ensemble-voting-averaging/
def getVote(models,Xs):
    print(np.where(models[0].predict(Xs[0])==0,-1,1))
    y_preds = np.vstack((np.where(models[i].predict(Xs[i])==0,-1,1) for i in range(3)))
    print(y_preds.shape)
    pred = np.sign(np.sum(y_preds,axis=0))
    print(pred.shape)
    pred = np.where(pred==-1,0,1)
    return pred