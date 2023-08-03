from asyncio.windows_events import NULL
from matplotlib import test
import numpy as np
from sklearn.linear_model import Perceptron

from trainingModule.inputData import inputData
from trainingModule.splitData import splitData
from trainingModule.evaluateModel import evaluateModel
from trainingModule.plotData import plotPCA, plotBar
from trainingModule.learning import fitModel,getVote
def conduct(nn_layer=(100,100), index2=0):
    #-----------------------------------------------------
    mode1= "neural network"
    mode2 = "k-neighbors"
    mode3 = "random forest" 
    r_num1=1800
    r_num2 = 0.001
    r_num3 = 2000
    fingerprint13 = "rdMHFP"
    fingerprint2  = "MACCSKey"
    solver="adam"
    activation = 'relu'
    #------------------------------------------------------
    fingerprints=["morgan", "RDKit", "MACCSKey", "rdMHFP", "Avalon"]
    #test_parameter = [10, 100, 1000, 10000, 100000]
    test_parameter1 = list(range(100,2001,100))
    test_parameter2 = list(range(1,10))
    test_parameter3 = list(range(10,10001,10))
    X1, y = inputData(fingerprint=fingerprint13)
    X1_train, X1_test, y1_train, y1_test = splitData(X1, y)
    X2, y = inputData(fingerprint=fingerprint2)
    X2_train, X2_test, y2_train, y2_test = splitData(X2, y)
    goodScore = 0
    goodScores = [0,0,0,0,0,0] #acc, F ,auc, i,j,k
    #accs=[]
    export_data = np.zeros((6,len(test_parameter1)**3))
    print(export_data.shape)
    index=0
    for i in test_parameter1:
        model1 = fitModel(X1_train, y1_train, mode = mode1, hidden_layer_sizes=nn_layer,random_state=i)
        for j in test_parameter2:
            model2 = fitModel(X2_train, y2_train, mode = mode2, gamma=r_num2, n_neighbors=j)
            for k in test_parameter3:
                model3 = fitModel(X1_train, y1_train, mode = mode3, random_state=k)
                y_test_pred = getVote([model1,model2,model3],[X1_test,X2_test,X1_test])

                acc, F , auc = evaluateModel(y1_test, y_test_pred, output=True)
                export_data[0,index]=i
                export_data[1,index]=j
                export_data[2,index]=k
                export_data[3,index]=acc
                export_data[4,index]=F
                export_data[5,index]=auc
                index+=1
                score = acc
                if score > goodScore:
                    goodScore = score
                    goodScores[0] = acc
                    goodScores[1] = F
                    goodScores[2] = auc
                    goodScores[3] = i
                    goodScores[4] = j
                    goodScores[5] = k
            print("-----------------------------------------------------------------------------")
            print("nn layer: ", nn_layer)
            print("most random_state:", goodScores[3])
            print("most acc:", goodScores[0])
            print("most F1:", goodScores[1])
            print("most auc:", goodScores[2])

    #plotBar(x_plot, np.asarray(accs),tick_label=test_parameter,xlabel="random state", ylabel="accuracy rate")
    np.savetxt("report_mixed_{}_{}_{}.csv".format(mode1,activation,index2),export_data.T,delimiter=",")
    print("finish")
#plotPCA(X_train, X_test, y_train, y_test, goodModel)
conduct()