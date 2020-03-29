from main import mglearn, np, plt, train_test_split

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer['data'], cancer['target'], random_state=0)

gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)

print("{:.3f}".format(gbrt.score(X_train, y_train)))
print("{:.3f}".format(gbrt.score(X_test, y_test)))
