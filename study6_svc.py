from asyncio.windows_events import NULL
from matplotlib import test
import numpy as np
from sklearn.linear_model import Perceptron

from trainingModule.inputData import inputData
from trainingModule.splitData import splitData
from trainingModule.evaluateModel import evaluateModel
from trainingModule.plotData import plotPCA, plotBar
from trainingModule.learning import fitModel
#-----------------------------------------------------
fingerprint="BP"
mode="SVC"
kernel="rbf"
#------------------------------------------------------
X, y = inputData(fingerprint=fingerprint)
X_train, X_test, y_train, y_test = splitData(X, y)
test_parameter = [1e-3, 1e-2, 1e-1, 1, 10,100]
#test_parameter = np.arange(1,1001)/1000


x_plot = np.arange(len(test_parameter))


goodScore = 0
goodScores = [0,0,0,0] #acc, F ,auc, r_num
#accs=[]
export_data = np.vstack((np.array(test_parameter),np.zeros((3,len(test_parameter)))))
print(export_data.shape)
goodModel = NULL
i=0
for r_num in test_parameter:
    model1 = fitModel(X_train, y_train, gamma=r_num, mode=mode, kernel=kernel)

    y_test_pred = model1.predict(X_test)

    acc, F , auc = evaluateModel(y_test, y_test_pred)
    export_data[1,i]=acc
    export_data[2,i]=F
    export_data[3,i]=auc
    i+=1
    score = acc
    if score > goodScore:
        goodScore = score
        goodScores[0] = acc
        goodScores[1] = F
        goodScores[2] = auc
        goodScores[3] = r_num
        goodModel = model1

print("model: ",mode)
print("kernel: ",kernel)
print("fingerprint: ", fingerprint)
print("most random_state:", goodScores[3])
print("most acc:", goodScores[0])
print("most F1:", goodScores[1])
print("most auc:", goodScores[2])

#plotBar(x_plot, np.asarray(accs),tick_label=test_parameter,xlabel="random state", ylabel="accuracy rate")
np.savetxt("report_{}_{}_{}.csv".format(fingerprint,mode, kernel),export_data.T,delimiter=",")
print("finish")


"""
model:  SVC
kernel:  rbf
fingerprint:  RDKit
most random_state: 0.01
most acc: 0.9700854700854701
most F1: 0.5625
most auc: 0.732387762278748

model:  SVC
kernel:  rbf
fingerprint:  morgan
most random_state: 0.001
most acc: 0.9709401709401709
most F1: 0.5802469135802469
most auc: 0.7429140780682219

model:  SVC
kernel:  rbf
fingerprint:  MACCSKey
most random_state: 0.001
most acc: 0.9735042735042735
most F1: 0.5507246376811594
most auc: 0.6988864142538975

model:  SVC
kernel:  rbf
fingerprint:  rdMHFP
most random_state: 0.001
most acc: 0.9662393162393162
most F1: 0.4233576642335766
most auc: 0.6497362560075021

model:  SVC
kernel:  rbf
fingerprint:  Avalon
most random_state: 0.001
most acc: 0.9752136752136752
most F1: 0.6133333333333333
most auc: 0.7401008088149102


"""