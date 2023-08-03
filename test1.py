import numpy as np
from sklearn.linear_model import Perceptron

from trainingModule.inputData import inputData
from trainingModule.splitData import splitData
from trainingModule.evaluateModel import evaluateModel
from trainingModule.learning import fitModel
#-----------------------------------------------------
mode="Perceptron"
r_num=1234
fingerprint = "morgan"
#------------------------------------------------------
X, y = inputData(fingerprint=fingerprint)
X_train, X_test, y_train, y_test = splitData(X, y)
model1 = fitModel(X_train, y_train, mode = mode, n_neighbors=r_num, random_state=r_num)
y_test_pred = model1.predict(X_test)

acc, F , auc = evaluateModel(y_test, y_test_pred, output=True)
print("-----------------------------------------------------------------------------")
print("model: ",mode)
print("fingerprint: ", fingerprint)
print("random state: ", r_num)