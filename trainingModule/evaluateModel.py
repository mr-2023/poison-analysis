from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score

def evaluateModel(y_test, y_test_pred, output=False):
    acc =  accuracy_score(y_test , y_test_pred)
    F = f1_score(y_test, y_test_pred)
    auc = roc_auc_score(y_test, y_test_pred)
    if output:
        print("Acc:", acc) #ACC:  正しく判別できたデータの数  /  全データ　　　　　　　　　　　　　　　　　　　　　　　　　　　　  0.9602564102564103
        print("F:", F)          #F  :  適合率(真陽性/陽性予測)と再現率(真陽性/実陽性)の調和平均　　　実係数βで調整               0.5079365079365079
        print("AUC:", auc)   #縦：　新陽性率　横：偽陽性率の時、偽陽性率を上げると真陽性率も上がる　このグラフの下側の面積の割合    0.7423865900832258
    return acc, F , auc
