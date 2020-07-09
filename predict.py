import pickle
import numpy as np

from config import *

def loadPickle(filepath):
    pickle_in = open(filepath, "rb")
    return pickle.load(pickle_in)

linear = loadPickle('stock_model.pickle')
test_data = loadPickle('test_data.pickle')

x_test = np.array(test_data.drop([predict], 1).drop(['Date'], 1))

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print('Prediction of', test_data['Date'].values[0], ':')
    print('$', round(predictions[x], 2))
    print('Actual: ')
    print('$', test_data[predict].values[0])
