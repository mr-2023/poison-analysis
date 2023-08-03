from asyncio.windows_events import NULL
from matplotlib import test
import numpy as np
from sklearn.linear_model import Perceptron

from trainingModule.inputData import inputData
from trainingModule.splitData import splitData
from trainingModule.evaluateModel import evaluateModel
from trainingModule.plotData import plotPCA, plotBar
from trainingModule.learning import fitModel
from trainingModule.failPicture import failPicture
#-----------------------------------------------------
mode="neural network"
r_num=1100
fingerprint = "rdMHFP"
#------------------------------------------------------
X, y = inputData(fingerprint=fingerprint)
X_train, X_test, y_train, y_test = splitData(X, y)
model1 = fitModel(X_train, y_train, mode = mode, n_neighbors=r_num,hidden_layer_sizes=(100,100),random_state=r_num)
y_test_pred = model1.predict(X_test)

acc, F , auc = evaluateModel(y_test, y_test_pred, output=True)
failPicture(y_test, y_test_pred)
print("-----------------------------------------------------------------------------")
print("model: ",mode)
print("fingerprint: ", fingerprint)
print("random state: ", r_num)