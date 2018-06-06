import numpy as np
import pandas as pd
from pandas2arff import pandas2arff
from scipy.io import arff
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import normalize

raw_data = pd.DataFrame(arff.loadarff("regression/cpu.small.arff")[0])

X = raw_data.iloc[:, :-1]
Y = raw_data.iloc[:, -1:]

X_norm = normalize(X, axis=0)

half_raw_num_columns = int(raw_data.shape[0]/2)

data_train = np.concatenate([X_norm[0:half_raw_num_columns],
                             Y.iloc[0:half_raw_num_columns]],
                            axis=1)

#pandas2arff(pd.DataFrame(data_train), "cpu.small_RQP_train.arff", wekaname="test")

X_gt_train = X_norm.iloc[half_raw_num_columns:]
Y_gt_train = Y.iloc[half_raw_num_columns:]
