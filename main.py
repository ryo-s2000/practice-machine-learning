# 共通import処理(データセット、モデルの読み込みは各自で)
# 鬱陶しいFutureWarningを無視
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import mglearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

__all__ = ['simplefilter', 'pd', 'np', 'mglearn', 'plt', 'train_test_split']

# いつもの
# X_train, X_test, y_train, y_test = train_test_split(random_state=)
# print("{}".format())