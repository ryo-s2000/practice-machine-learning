from sklearn.datasets import load_iris
import pandas as pd
import mglearn
import matplotlib.pyplot as plt
iris_dataset = load_iris()

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0
)

df = pd.DataFrame(data=iris_dataset.data, columns=iris_dataset.feature_names)

pd.plotting.scatter_matrix(df,figsize=(8,8), marker='o', hist_kwds={'bins':20}, s=60, alpha=.8, cmap=mglearn.cm3)
plt.show()
