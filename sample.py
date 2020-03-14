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

# key取得
print("{}".format(iris_dataset.keys()))

# 計測済データ
print("{}".format(iris_dataset['data']))

# 目標番号
print("{}".format(iris_dataset['target']))

# 目標名
print("{}".format(iris_dataset['target_names']))

# 特徴名
print("{}".format(iris_dataset['feature_names']))

# データ格納場所
print("{}".format(iris_dataset['filename']))

# 説明
print("{}".format(iris_dataset['DESCR']))
