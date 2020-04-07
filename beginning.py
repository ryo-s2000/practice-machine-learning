from main import mglearn, train_test_split, plt, np, pd

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

X, y = mglearn.datasets.make_wave(n_samples=100)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
reg = DecisionTreeRegressor(min_samples_split=3).fit(X, y)
plt.plot(line, reg.predict(line), label="linear regression")

plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel("R")
plt.xlabel("I")
plt.legend(loc='best')

bins = np.linspace(-3, 3, 11)
print("{}".format(bins))

which_bin = np.digitize(X, bins=bins)
print(X[:5])
print(which_bin[:5])

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
encoder.fit(which_bin)
X_binned = encoder.transform(which_bin)
print(X_binned[:5])

print(X[:,0])

# plt.show()
