from asyncio.windows_events import NULL
from matplotlib import test
import numpy as np
from sklearn.linear_model import Perceptron

from trainingModule.inputData import inputData
from trainingModule.splitData import splitData, checkTrue
from trainingModule.evaluateModel import evaluateModel
from trainingModule.learning import fitModel, fitModelFailed, getVote
from trainingModule.failPicture import failPicture
def conduct(fingerprint2="Avalon"):
    #-----------------------------------------------------
    mode1= "neural network"
    mode2 = "k-neighbors"
    mode3 = "random forest" 
    r_num1=1800
    r_num2 = 0.001
    r_num3 = 2000
    fingerprint13 = "rdMHFP"
    #------------------------------------------------------
    X1, y = inputData(fingerprint=fingerprint13)
    X1_train, X1_test, y1_train, y1_test = splitData(X1, y)
    X2, y = inputData(fingerprint=fingerprint2)
    X2_train, X2_test, y2_train, y2_test = splitData(X2, y)
    checkTrue(y1_train,y2_train)
    model1 = fitModel(X1_train, y1_train, mode = mode1, hidden_layer_sizes=(100,100),random_state=r_num1)
    y_pred= model1.predict(X1_train)
    model2 = fitModelFailed(X2_train, y2_train, y_pred=y_pred,mode = mode2, gamma=r_num2, n_neighbors=2)
    y_pred = model2.predict(X2_train)
    model3 = fitModelFailed(X1_train, y1_train, y_pred=y_pred,mode = mode3, random_state=r_num3)
    print("y_pred:",model1.predict(X1))
    y_test_pred = getVote([model1,model2,model3],[X1_test,X2_test,X1_test])
    print("y_pred:",len(y_test_pred), "y:", len(y1_test))
    acc, F , auc = evaluateModel(y1_test, y_test_pred, output=True)
    failPicture(y1_test, y_test_pred)
    print("----------------------------------------------------------------------")
conduct(fingerprint2="MACCSKey")
"""
fingerprints = ["morgan","RDKit","MACCSKey", "rdMHFP", "Avalon"]
for fingerprint in fingerprints:
    conduct(fingerprint2=fingerprint)
"""
"""
NN SVC RF
Acc: 0.976068376068376
F: 0.631578947368421
AUC: 0.7506271246043841

(失敗補正)
y_pred: 2340 y: 2340
Acc: 0.9769230769230769
F: 0.6399999999999999
AUC: 0.751072558902825

NN k-neighbor(morgan) RF
Acc: 0.9764957264957265
F: 0.6308724832214765
AUC: 0.7458094010080881
失敗補正
y_pred: 2340 y: 2340
Acc: 0.9739316239316239
F: 0.6257668711656442
AUC: 0.7646348610948306
"""