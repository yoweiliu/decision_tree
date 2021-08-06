import graphviz
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree

df = pd.read_csv("raw_data", delim_whitespace=True)
# print(df)

## 手動編碼不同特徵
# label_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
# #將編碼後的label map存至df_data['Species']中。
# df['Class'] = df['Species'].map(label_map)

features = list(df.columns[:4])
x, y= df[features], df["能不能打羽球"]

classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(x, y)

## 畫決策樹狀圖  (圖片中文亂碼未修正)
plt.figure(figsize=(6, 6))
tree.plot_tree(classifier,feature_names=features)  
# plt.show()

# digraph g {
#      node[fontname = "Microsoft JhengHei"];
#      "中文" -> "英文";
# }

## 將決策樹的dot_data檔案格式保存
dot_data = tree.export_graphviz(classifier, out_file=None,
                                feature_names=features,
                                class_names=["不能打羽球", "可以打羽球"],
                                filled=True, rounded=True, leaves_parallel=True)
""" 
.replace('helvetica','"Microsoft YaHei"'), encoding='utf-8'
上述 解決中文亂碼
"""
graph = graphviz.Source(dot_data.replace('helvetica','"Microsoft YaHei"'), encoding='utf-8') 
graph.render("badminton") #檔名
graph.view()


## 顯示各特徵重要程度   (dataframe 對齊未修正)
d = {'feature': features, 'importance': classifier.feature_importances_}
features_level = pd.DataFrame(d).sort_values(by=['importance'], ascending=False)
print('   ' + '特徵重要程度: \n',features_level)

# result = classifier.predict([[2.3, 9.3, 5.4, 9.1]])
# print(result)
