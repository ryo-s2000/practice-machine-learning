from main import mglearn, plt, train_test_split

from sklearn.neighbors import KNeighborsClassifier

# # 学習用データを呼び出し(モジュール無いに事前に準備されている)
# X, y = mglearn.datasets.make_forge()

# # 試しにデータを与えて、分類をしてみる
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# # クラス分類機を訓練
# clf = KNeighborsClassifier(n_neighbors=3)

# # 実行
# clf.fit(X_train, y_train)

# # 結果、正答率
# print("Test set prediction: {}".format(clf.predict(X_test)))
# print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))

# # 境界線を表示
# fig, axes = plt.subplots(1,3, figsize=(10, 3))
# for n_neighbors, ax in zip([1, 3, 9], axes):
#     clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X,y)
#     mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
#     mglearn.discrete_scatter(X[:, 0], X[:, 1], y,ax=ax)
#     ax.set_title("{} neighboor(s)".format(n_neighbors))
#     ax.set_xlabel("feature 0")
#     ax.set_ylabel("feature 0")
# axes[0].legend(loc=3)
# plt.show()

# いくつ点を与えれば正答率が高くなるのか(k-最近傍法)
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer['data'], cancer['target'], stratify=cancer['target'], random_state=66)
training_accuracy = []
test_accuracy = []
neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)

    training_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()
