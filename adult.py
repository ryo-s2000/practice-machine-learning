from main import mglearn, train_test_split, plt, np, pd

import pandas as os
import os
from IPython.core.display import display

adult_path = os.path.join(mglearn.datasets.DATA_PATH, 'adult.data')

data = pd.read_csv(
    adult_path, header=None, index_col=False,
    names=['age', 'workclass', 'fnlwget', 'education', 'education-num', 'm', 'occupation', 
    'r', 're', 'gender', 'cap', 'los', 'hours-per-week', 'native', 'income']
)

data = data[['age', 'workclass', 'education', 'education-num', 'gender', 'hours-per-week', 'occupation', 'income']]

print(data.gender.value_counts())
print(data.columns)
data_dummies = pd.get_dummies(data)
print(data_dummies.columns)

features = data_dummies.loc[:, 'age':'occupation_ Transport-moving']
X = features.values
y = data_dummies['income_ >50K'].values
print("{}{}".format(X.shape, y.shape))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print("{:2f}".format(logreg.score(X_test, y_test)))

display(data.head())
