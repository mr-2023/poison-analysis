from asyncio.windows_events import NULL
from matplotlib import test
import numpy as np
from sklearn.linear_model import Perceptron

from trainingModule.inputData import inputData
from trainingModule.splitData import splitData
from trainingModule.evaluateModel import evaluateModel
from trainingModule.plotData import plotPCA, plotBar
from trainingModule.learning import fitModel
def conduct(nn_layer=(100,), index=0):
    #-----------------------------------------------------
    mode="neural network"
    solver="adam"
    activation = 'relu'
    #------------------------------------------------------
    fingerprints=["morgan", "RDKit", "MACCSKey", "rdMHFP", "Avalon"]
    for fingerprint in fingerprints:
        X, y = inputData(fingerprint=fingerprint)
        X_train, X_test, y_train, y_test = splitData(X, y)
        #test_parameter = [10, 100, 1000, 10000, 100000]
        test_parameter = list(range(100,2001,100))
        if mode == "k-neighbors":
            test_parameter = list(range(1,10))


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
                model1 = fitModel(X_train, y_train, random_state=r_num, mode=mode,solver=solver,hidden_layer_sizes=nn_layer, activation=activation)

            y_test_pred = model1.predict(X_test)

            acc, F , auc = evaluateModel(y_test, y_test_pred, output=True)
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
        print("-----------------------------------------------------------------------------")
        print("model: ",mode)
        print("fingerprint: ", fingerprint)
        print("nn layer: ", nn_layer)
        print("most random_state:", goodScores[3])
        print("most acc:", goodScores[0])
        print("most F1:", goodScores[1])
        print("most auc:", goodScores[2])

        #plotBar(x_plot, np.asarray(accs),tick_label=test_parameter,xlabel="random state", ylabel="accuracy rate")
        np.savetxt("report_{}_{}_{}_{}.csv".format(fingerprint,mode,activation,index),export_data.T,delimiter=",")
        print("finish")
    #plotPCA(X_train, X_test, y_train, y_test, goodModel)

nn_layers = [(100,), (80,), (300,), (100,10),(200,100),(100,100)]
nn_layers = [(100,10),(200,100),(100,100)]
conduct(nn_layer=(100,90),index=7)
i=3

for taple in nn_layers:
    if i==6:
        i+=1
    conduct(taple,index=i)
    i+=1

"""
(100,)
model:  neural network
fingerprint:  morgan
most random_state: 900
most acc: 0.9683760683760684
most F1: 0.5487804878048781
most auc: 0.7314968936818661


model:  neural network
fingerprint:  RDKit
most random_state: 900
most acc: 0.9730769230769231
most F1: 0.6134969325153375
most auc: 0.7591489860508732

model:  neural network ########################
fingerprint:  MACCSKey
most random_state: 1000
most acc: 0.9735042735042735
most F1: 0.5974025974025974
most auc: 0.7392099402180283

model:  neural network
fingerprint:  rdMHFP
most random_state: 500
most acc: 0.9722222222222222
most F1: 0.5962732919254659
most auc: 0.7486226702613996

model:  neural network
fingerprint:  Avalon
most random_state: 600
most acc: 0.9713675213675214
most F1: 0.5988023952095808
most auc: 0.7582581174539913
"""

"""
(100,10)
model:  neural network
fingerprint:  morgan
most random_state: 600
most acc: 0.9696581196581197
most F1: 0.5534591194968552
most auc: 0.7271246043840114

model:  neural network
fingerprint:  RDKit
most random_state: 1900
most acc: 0.9717948717948718
most F1: 0.6071428571428571
most auc: 0.7635212753487282

model:  neural network
fingerprint:  MACCSKey
most random_state: 1800
most acc: 0.9717948717948718
most F1: 0.5925925925925926
most auc: 0.7483999531121791
#################################
model:  neural network
fingerprint:  rdMHFP
most random_state: 1800
most acc: 0.9756410256410256
most F1: 0.6013986013986015
most auc: 0.7252022037275818

model:  neural network
fingerprint:  Avalon
most random_state: 1200
most acc: 0.9713675213675214
most F1: 0.5838509316770186
most auc: 0.7431367952174422

"""

"""
(200,100)
model:  neural network
fingerprint:  morgan
most random_state: 1100
most acc: 0.9713675213675214
most F1: 0.5838509316770186
most auc: 0.7431367952174422

model:  neural network
fingerprint:  RDKit
most random_state: 900
most acc: 0.9735042735042735
most F1: 0.6075949367088607
most auc: 0.7492908217090611

model:  neural network
fingerprint:  MACCSKey
most random_state: 1900
most acc: 0.9722222222222222
most F1: 0.5962732919254659
most auc: 0.7486226702613996

model:  neural network ############################
fingerprint:  rdMHFP
most random_state: 300
most acc: 0.9743589743589743
most F1: 0.5833333333333334
most auc: 0.7194936115344039

model:  neural network
fingerprint:  Avalon
most random_state: 300
most acc: 0.9722222222222222
most F1: 0.6153846153846154
most auc: 0.768784433243465

"""

"""
(100,100)
model:  neural network
fingerprint:  morgan
most random_state: 1300
most acc: 0.9700854700854701
most F1: 0.5679012345679012
most auc: 0.7374282030242644

model:  neural network
fingerprint:  RDKit
most random_state: 100
most acc: 0.9739316239316239
most F1: 0.6211180124223601
most auc: 0.7595944203493142

model:  neural network
fingerprint:  MACCSKey
most random_state: 1800
most acc: 0.9726495726495726
most F1: 0.5949367088607594
most auc: 0.7438049466651037
######################
model:  neural network
fingerprint:  rdMHFP
most random_state: 1100
most acc: 0.976068376068376
most F1: 0.6455696202531644
most auc: 0.765748446840933
finish

model:  neural network
fingerprint:  Avalon
most random_state: 700
most acc: 0.9722222222222222
most F1: 0.6060606060606061
most auc: 0.7587035517524322
"""

"""
(300,)
model:  neural network
fingerprint:  morgan
most random_state: 2000
most acc: 0.9696581196581197
most F1: 0.5644171779141104
most auc: 0.7372054858750439
#####################
model:  neural network
fingerprint:  RDKit
most random_state: 300
most acc: 0.9730769230769231
most F1: 0.6181818181818182
most auc: 0.7641894267963896

model:  neural network
fingerprint:  MACCSKey
most random_state: 1800
most acc: 0.9726495726495726
most F1: 0.5949367088607594
most auc: 0.7438049466651037

model:  neural network
fingerprint:  rdMHFP
most random_state: 100
most acc: 0.9722222222222222
most F1: 0.5517241379310345
most auc: 0.7082991442972688

model:  neural network
fingerprint:  Avalon
most random_state: 1700
most acc: 0.9726495726495726
most F1: 0.6097560975609756
most auc: 0.7589262689016527
"""

"""
(80,)
model:  neural network
fingerprint:  morgan
most random_state: 200
most acc: 0.967948717948718
most F1: 0.5508982035928144
most auc: 0.736314617278162
finish

model:  neural network
fingerprint:  RDKit
most random_state: 1400
most acc: 0.9730769230769231
most F1: 0.5987261146496815
most auc: 0.7440276638143243
finish

model:  neural network
fingerprint:  MACCSKey
most random_state: 1600
most acc: 0.9717948717948718
most F1: 0.5875
most auc: 0.7433595123666628
finish
###########
model:  neural network
fingerprint:  rdMHFP
most random_state: 1800
most acc: 0.976068376068376
most F1: 0.6056338028169014
most auc: 0.7254249208768022
finish

model:  neural network
fingerprint:  Avalon
most random_state: 1300
most acc: 0.9726495726495726
most F1: 0.6
most auc: 0.7488453874106201
finish
"""

"""
model:  neural network
fingerprint:  morgan
nn layer:  (100, 80)
most random_state: 800
most acc: 0.9696581196581197
most F1: 0.5696969696969696
most auc: 0.7422459266205604

model:  neural network
fingerprint:  RDKit
nn layer:  (100, 80)
most random_state: 1100
most acc: 0.9743589743589743
most F1: 0.625
most auc: 0.7598171374985347
finish

model:  neural network
fingerprint:  MACCSKey
nn layer:  (100, 80)
most random_state: 2000
most acc: 0.9726495726495726
most F1: 0.6
most auc: 0.7488453874106201

model:  neural network
fingerprint:  rdMHFP
nn layer:  (100, 80)
most random_state: 1900
most acc: 0.9764957264957265
most F1: 0.6153846153846154
most auc: 0.7306880787715391
finish

model:  neural network
fingerprint:  Avalon
nn layer:  (100, 80)
most random_state: 2000
most acc: 0.9713675213675214
most F1: 0.6035502958579881
most auc: 0.7632985581995076
"""

"""
model:  neural network
fingerprint:  morgan
nn layer:  (100, 80, 60)
most random_state: 1000
most acc: 0.9709401709401709
most F1: 0.5750000000000001
most auc: 0.7378736373227054
finish

model:  neural network
fingerprint:  RDKit
nn layer:  (100, 80, 60)
most random_state: 1900
most acc: 0.9717948717948718
most F1: 0.5975609756097561
most auc: 0.7534403938576956

model:  neural network
fingerprint:  MACCSKey
nn layer:  (100, 80, 60)
most random_state: 200
most acc: 0.9717948717948718
most F1: 0.5714285714285715
most auc: 0.7282381901301138

model:  neural network
fingerprint:  rdMHFP
nn layer:  (100, 80, 60)
most random_state: 1800
most acc: 0.976068376068376
most F1: 0.6056338028169014
most auc: 0.7254249208768022

model:  neural network
fingerprint:  Avalon
nn layer:  (100, 80, 60)
most random_state: 1200
most acc: 0.9709401709401709
most F1: 0.5903614457831325
most auc: 0.7529949595592544
"""




#----------------------------------------------------------------------
"""
model:  random forest
fingerprint:  morgan
nn layer:  (100, 80)
most random_state: 1100
most acc: 0.9735042735042735
most F1: 0.6219512195121951
most auc: 0.7644121439456101

model:  random forest
fingerprint:  RDKit
nn layer:  (100, 80)
most random_state: 1500
most acc: 0.9735042735042735
most F1: 0.6172839506172839
most auc: 0.7593717032000937
finish

model:  random forest
fingerprint:  MACCSKey
nn layer:  (100, 80)
most random_state: 500
most acc: 0.9735042735042735
most F1: 0.6125
most auc: 0.7543312624545774

model:  random forest
fingerprint:  rdMHFP
nn layer:  (100, 80)
most random_state: 2000
most acc: 0.9739316239316239
most F1: 0.6211180124223601
most auc: 0.7595944203493142
finish

model:  random forest
fingerprint:  Avalon
nn layer:  (100, 80)
most random_state: 2000
most acc: 0.9739316239316239
most F1: 0.6257668711656442
most auc: 0.7646348610948306
"""