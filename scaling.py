from main import mglearn, train_test_split, plt, np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

cancer = load_breast_cancer()

X_train, X_test ,y_train, y_test = train_test_split(cancer['data'], cancer['target'], random_state=1)

print(X_train.shape)
print(X_test.shape)

scaler = MinMaxScaler()
scaler.fit(X_train)
MinMaxScaler(copy=True, feature_range=(0, 1))

X_train_scaled = scaler.transform(X_train)
print("{}".format(X_train_scaled.shape))
print("{}".format(X_train.min(axis=0)))
print("{}".format(X_train.max(axis=0)))
print("{}".format(X_train_scaled.min(axis=0)))
print("{}".format(X_train_scaled.max(axis=0)))

X_test_scaled = scaler.transform(X_test)
print("{}".format(X_test_scaled.min(axis=0)))
print("{}".format(X_test_scaled.max(axis=0)))

# from sklearn.datasets import make_blobs
# X, _ = make_blobs(n_samples=50, centers=5, random_state=4, cluster_std=2)
# X_train, X_test ,y_train, y_test = train_test_split(X, random_state=1, test_size=.1)
# fig, axes = plt.subplots(1, 3, figssize=(13, 4))
# axes[0].scatter(X_train[:, 0], X_train[:, 1], c=mglearn.cm2(0), label="Training set", s=60)
# axes[0].scatter(X_test[:, 0], X_test[:, 1], marker="^", c=mglearn.cm2(1), label="Test set", s=60)
# axes[0].legend(loc='upper left')
# axes[0].set_title('Original Data')

# scaler = MinMaxScaler()
# scaler.fit(X_train)
# X_train_scaled = scaler.transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=mglearn.cm2(0), lebel="Training set", s=60)
# axes[1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], marker="^", c=mglearn.cm2(0), lebel="Training set", s=60)
# axes[1].set_title("Scaled Data")

# mglearn.plots.plot_scaling()
# plt.show()

from sklearn.svm import SVC
svm = SVC(C=100)
svm.fit(X_train, y_train)
print("{}".format(svm.score(X_train, y_train)))
