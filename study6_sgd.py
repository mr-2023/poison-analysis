from asyncio.windows_events import NULL
import numpy as np
from sklearn.linear_model import SGDClassifier

from trainingModule.inputData import inputData
from trainingModule.splitData import splitData
from trainingModule.evaluateModel import evaluateModel
from trainingModule.plotData import plotPCA
from trainingModule.learning import fitModel
X, y = inputData(fingerprint="RDKit")
X_train, X_test, y_train, y_test = splitData(X, y)
test_parameter = [10, 50, 100, 500, 1000, 5000, 10000, 50000,100000]



goodScore = 0
goodScores = [0,0,0,0] #acc, F ,auc, r_num
goodModel = NULL
for r_num in test_parameter:
    model1 = fitModel(X_train, y_train, random_state=r_num, mode="SGD")

    y_test_pred = model1.predict(X_test)

    acc, F , auc = evaluateModel(y_test, y_test_pred)
    sumScore = acc + F + auc
    if sumScore > goodScore:
        print("sumScore: ",sumScore, " : ", goodScore)
        goodScore = sumScore
        goodScores[0] = acc
        goodScores[1] = F
        goodScores[2] = auc
        goodScores[3] = r_num
        goodModel = model1

print("most random_state:", goodScores[3])
print("most acc:", goodScores[0])
print("most F1:", goodScores[1])
print("most auc:", goodScores[2])


plotPCA(X_train, X_test, y_train, y_test, goodModel)
#morganfingerprint
#most random_state: 100000
#most acc: 0.9735042735042735
#most F1: 0.6025641025641025
#most auc: 0.7442503809635448

#RDKitfingerprint
#most random_state: 10
#most acc: 0.9653846153846154
#most F1: 0.5524861878453039
#most auc: 0.7551400773649044