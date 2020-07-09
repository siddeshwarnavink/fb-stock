import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import pickle

from config import *

data = pd.read_csv('FB.csv')
last_row = len(data)
data.drop(data.tail(1).index,inplace=True)

# Saveing the last row as testing data
with open('test_data.pickle', 'wb') as f:
    pickle.dump(data.tail(1), f)

x = np.array(data.drop([predict], 1).drop(['Date'], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    x, y, test_size=0.1)

# Saveing the best of 30 models
best = 0
for _ in range(30):
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)

    acc = linear.score(x_test, y_test)
    print("Accuracy: ", acc)

    if acc > best:
        best = acc
        with open('stock_model.pickle', 'wb') as f:
            pickle.dump(linear, f)
