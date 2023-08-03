#https://qiita.com/keimoriyama/items/f93f07514d98704e3810

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
#test_idx: 全データのうち,テストデータの範囲
def plot_decision_regions(X,y,classifier,test_idx,resolution = 0.02, pca=None):
    #マーカーとカラーマップの用意
    markers = ('s','x')
    colors = ('red','blue')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    #決定領域のプロット
    x1_min,x1_max = X[:,0].min() - 1,X[:,0].max() + 1
    x2_min,x2_max = X[:,1].min() - 1,X[:,1].max() + 1
    #グリッドポイントの作成
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),
                         np.arange(x2_min,x2_max,resolution))
    print("xx1: ", xx1.shape, " ,xx2: ", xx2.shape)
    x_inv = np.array([xx1.ravel(), xx2.ravel()]).T
    print("x_inv: ", x_inv.shape)
    if pca is not None:
        x_inv = pca.inverse_transform(x_inv)
        print("x_inv: ", x_inv.shape)
    #特徴量を一次元配列に変換して予測を実行する
    Z = classifier.predict(x_inv)
    #予測結果をグリッドポイントのデータサイズに変換
    Z = Z.reshape(xx1.shape)
    #グリッドポイントの等高線のプロット
    plt.contourf(xx1,xx2,Z,alpha = 0.4,cmap = cmap)
    #軸の範囲の設定
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())
    #クラスごとにサンプルをプロット
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0],y=X[y == cl,1],alpha=0.8,c = colors[idx],marker=markers[idx],label=cl)
    #テストサンプルを目立たせる
    if test_idx:
        X_test,y_test = X[test_idx,:],y[test_idx]
        plt.scatter(X_test[:,0],X_test[:,1],linewidths=1,marker='o',s = 10,label = 'test set',alpha=0.1)

    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    #凡例の設定
    plt.legend(loc = 'upper left')
    #グラフの表示
    plt.show()


