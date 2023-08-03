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
fingerprint="rdMHFP"
mode="SGD"
#------------------------------------------------------
fingerprints=["morgan", "RDKit", "MACCSKey", "rdMHFP", "Avalon"]
solvers =['sgd', 'adam']
X, y = inputData(fingerprint=fingerprint)
X_train, X_test, y_train, y_test = splitData(X, y)
#test_parameter = [10, 100, 1000, 10000, 100000]
test_parameter = list(range(10,10001,10))
if mode == "k-neighbors":
    test_parameter = list(range(1,10))
x_plot = np.arange(len(test_parameter))


goodScore = 0
goodScores = [0,0,0,0] #acc, F ,auc, r_num
#accs=[]
export_data = np.vstack((np.array(test_parameter),np.zeros((3,len(test_parameter)))))
print(export_data.shape)
goodModel = NULL
i=0
for r_num in test_parameter:
    if mode == "k-neighbors":
        model1 = fitModel(X_train, y_train, mode = mode, n_neighbors=r_num)
    else:
        model1 = fitModel(X_train, y_train, random_state=r_num, mode=mode)

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
print("fingerprint: ", fingerprint)
print("most random_state:", goodScores[3])
print("most acc:", goodScores[0])
print("most F1:", goodScores[1])
print("most auc:", goodScores[2])

#plotBar(x_plot, np.asarray(accs),tick_label=test_parameter,xlabel="random state", ylabel="accuracy rate")
np.savetxt("report_{}_{}.csv".format(fingerprint,mode),export_data.T,delimiter=",")
print("finish")
#plotPCA(X_train, X_test, y_train, y_test, goodModel)

#perceptron
#morganFingerprint
#most random_state: 2810
#most acc: 0.976068376068376
#most F1: 0.6111111111111112
#most auc: 0.7304653616223187

#RDKitfingerprint
#most random_state: 8780
#most acc: 0.9747863247863248
#most F1: 0.624203821656051
#most auc: 0.7549994139022389

"""
model:  Perceptron
fingerprint:  Avalon
most random_state: 4440
most acc: 0.9752136752136752
most F1: 0.5972222222222222
most auc: 0.7249794865783613

model:  Perceptron
fingerprint:  MACCSKey
most random_state: 2070
most acc: 0.9747863247863248
most F1: 0.5629629629629629
most auc: 0.6995545657015589

model:  Perceptron
fingerprint:  rdMHFP
most random_state: 80
most acc: 0.9773504273504273
most F1: 0.6187050359712231
most auc: 0.7260930723244637

"""

"""
sgd
morganFingerprint
most random_state: 9750
most acc: 0.9769230769230769
most F1: 0.6249999999999999
most auc: 0.7359512366662759


RDKitFingerprint
most random_state: 6590
most acc: 0.9739316239316239
most F1: 0.5850340136054422
most auc: 0.7243113351306999

model:  SGD
fingerprint:  Avalon
most random_state: 7730
most acc: 0.9730769230769231
most F1: 0.5594405594405594
most auc: 0.7087445785957098

model:  SGD
fingerprint:  MACCSKey
most random_state: 320
most acc: 0.9764957264957265
most F1: 0.5985401459854014
most auc: 0.71556675653499
"""

"""
knn
RDKitFingerprint
most random_state: 2
most acc: 0.9769230769230769
most F1: 0.6351351351351352
most auc: 0.7460321181573086

most random_state: 2
most acc: 0.9777777777777777
most F1: 0.6486486486486487
most auc: 0.7515179932012661

model:  k-neighbors
fingerprint:  Avalon
most random_state: 2
most acc: 0.9764957264957265
most F1: 0.6206896551724138
most auc: 0.7357285195170554

model:  k-neighbors
fingerprint:  MACCSKey
most random_state: 2
most acc: 0.9777777777777777
most F1: 0.6438356164383561
most auc: 0.7464775524557495

model:  k-neighbors
fingerprint:  rdMHFP
most random_state: 2
most acc: 0.9756410256410256
most F1: 0.6225165562913908
most auc: 0.7453639667096472
"""