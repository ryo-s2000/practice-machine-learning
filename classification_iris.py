# 鬱陶しいFutureWarningを無視
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import mglearn
import matplotlib.pyplot as plt

# 学習用データを呼び出し(モジュール無いに事前に準備されている)
iris_dataset = load_iris()

# データセットを作成(テスト用と学習用に分ける)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0
)

# データをプロット
df = pd.DataFrame(data=iris_dataset['data'], columns=iris_dataset['feature_names'])
pd.plotting.scatter_matrix(df,figsize=(8,8), marker='o', hist_kwds={'bins':20}, s=60, alpha=.8, cmap=mglearn.cm3)
plt.show()

# 試しにデータを与えて、分類をしてみる
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, Y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=1, p=2, weights='uniform')

# 新しく分類するデータを作成
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))

# 結果表示
prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted terget name: {}".format(
    iris_dataset['target_names'][prediction]
))

# 精度のチェック 
y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))
print("Test set score: {:.2f}".format(np.mean(y_pred == Y_test)))
