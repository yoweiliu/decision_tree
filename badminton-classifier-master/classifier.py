import pandas as pd
from sklearn import tree
import graphviz

df = pd.read_csv("raw_data", delim_whitespace=True)

print(df)

features = list(df.columns[:4])
X = df[features]
Y = df["能不能打羽球"]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

print(clf.feature_importances_)

dot_data = tree.export_graphviz(clf, out_file=None,
                     feature_names=features,
                     class_names=["不能打羽球", "可以打羽球"],
                     filled=True, rounded=True, leaves_parallel=True)
graph = graphviz.Source(dot_data)
graph.render("badminton")

result = clf.predict([[5.3, 6.4, 5.4, 9.1]])
print(result)
