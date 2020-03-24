from main import mglearn, train_test_split, plt, np

from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC

X, y = make_blobs(random_state=42)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)

linear_svm = LinearSVC().fit(X, y)

# 傾き(ベクトル)
print("Coefficient shape: ", linear_svm.coef_.shape)
print("Coefficient shape: ", linear_svm.coef_)
# 切片
print("Intercept shape: ", linear_svm.intercept_.shape)
print("Intercept shape: ", linear_svm.intercept_)

line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, ['b', 'r', 'g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
plt.ylim(-10, 15)
plt.xlim(-10, 8)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line class 0', 'Line class 1', 'Line class 1', 'Line class 2'], loc=(1.01, 0.3))

mglearn.plots.plot_2d_classification(linear_svm, X, fill=True, alpha=.7)

plt.show()
