from main import mglearn, train_test_split

# mglearn.plots.plot_animal_tree()

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer['data'], cancer['target'], stratify=cancer['target'], random_state=42)
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)
print("{:.3f}".format(tree.score(X_train, y_train)))
print("{:.3f}".format(tree.score(X_test, y_test)))

export_graphviz(tree, out_file="tree.dot", class_names=["malignant", "benign"], feature_names=cancer["feature_names"], impurity=False, filled=True)

import graphviz
with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)
