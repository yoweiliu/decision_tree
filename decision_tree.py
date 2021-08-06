# https://reurl.cc/kZQ05b

import graphviz  # 繪圖(可視圖)
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap  # 繪圖(邊界圖)
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# (dataframe 對齊未修正)
df = pd.read_csv("raw_data", delim_whitespace=True)


features = list(df.columns[:4])
x, y = df[features], df["能不能打羽球"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=.2, random_state=42)  # 切割訓練集與測試集(random_state確保每次訓練結果都一樣，不確定如何設)

# print('Training data shape:',x_train.shape)   #印出訓練集 資料數、維度
# print('Testing data shape:',x_test.shape)



""" 
繪製決策邊界
以2維特徵繪圖，會是平面的形式
以3維特徵，則是立體三度空間的形式
"""


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=[cmap(idx)], marker=markers[idx], label=cl)
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='',
                    alpha=1.0, linewidth=1, marker='o',
                    s=55, label='test set')
    plt.show()

classifier = tree.DecisionTreeClassifier()  # 建立模型，gini/entropy，預設為gini
classifier = classifier.fit(x_train, y_train)  # 使用訓練資料訓練模型
predicted = classifier.predict(x_train)  # 使用訓練資料，預測分類


## 畫決策樹狀圖  
plt.figure(figsize=(6, 6))
tree.plot_tree(classifier,feature_names=features)

# # 顯示各特徵重要程度   (dataframe 對齊未修正)
d = {'feature': features, 'importance': classifier.feature_importances_}
features_level = pd.DataFrame(d).sort_values(
    by=['importance'], ascending=False)

print('   ' + '特徵重要程度: \n',features_level)


# 將決策樹的dot_data檔案格式保存
dot_data = tree.export_graphviz(classifier, out_file=None,
                                feature_names=features,
                                class_names=["不能打羽球", "可以打羽球"],
                                filled=True, rounded=True, leaves_parallel=True)
""" 
.replace('helvetica','"Microsoft YaHei"'), encoding='utf-8'
上述 解決中文亂碼
"""
graph = graphviz.Source(dot_data.replace('helvetica','"Microsoft YaHei"'), encoding='utf-8') #格式
graph.render("badminton") #檔名 存成pdf檔
graph.view()

# PCA可視化降維，將4個特徵只保留2個
# pca = PCA(n_components=2, iterated_power=1)  # n_components保留數
# train_reduced = pca.fit_transform(x_train)
# test_reduced = pca.transform(x_test)

# 訓練集預測
# plot_decision_regions(train_reduced, y_train, classifier)  # 決策邊界，2維較方便做可視化
# print('train set accurancy: ', classifier.score(train_reduced, y_train))
# # 測試集預測
# plot_decision_regions(test_reduced, y_test, classifier)
# print('test set accurancy: ', classifier.score(test_reduced, y_test))

