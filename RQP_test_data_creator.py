import numpy as np
import pandas as pd
from pandas2arff import pandas2arff
from scipy.io import arff
from scipy import optimize
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV

raw_data = pd.DataFrame(arff.loadarff("regression/cpu.small.arff")[0])

X = raw_data.iloc[:, :-1]
Y = raw_data.iloc[:, -1:]

X_norm = normalize(X, axis=0)

N_half = int(raw_data.shape[0]/2)

# Second half of data provides training data for synthetic ground truth
X_gt_train = X_norm[N_half:]
Y_gt_train = np.ravel(Y.iloc[N_half:])

# Default gaussian process regressor as synthetic ground truth
synth_ground_truth = GaussianProcessRegressor()

# MLP regressor with alpha-parameter determined by grid search cross validation (performed significantly worse than GP)
# alphas = {'alpha': 10.0 ** -np.arange(1, 7)}
# synth_ground_truth = GridSearchCV(MLPRegressor(), alphas)

synth_ground_truth.fit(X_gt_train, Y_gt_train)

# First half of data with synthetic ground truth Y-values as training data for RQP
X_RQP_train = X_norm[0:N_half]
Y_RQP_train = np.reshape(synth_ground_truth.predict(X_RQP_train), (N_half, 1))
data_train = np.concatenate([X_RQP_train, Y_RQP_train], axis=1)
#pandas2arff(pd.DataFrame(data_train), "cpu.small_RQP_train.arff", wekaname="test")

print("Score on original RQP training data:", synth_ground_truth.score(X_RQP_train, np.ravel(Y.iloc[0:N_half])))

# Generate N_half many interval-valued test data points for RQP
# by optimizing on random intervals on synthetic ground truth
data_test = []
num_features = X_RQP_train.shape[1]
#for i in range(N_half):

feature_intervals = []
initial_guess = []
for feature in range(num_features):
    x1 = np.random.random_sample()
    x2 = np.random.random_sample()
    lower_bound = np.min([x1, x2])
    upper_bound = np.max([x1, x2])
    feature_intervals.append( (lower_bound, upper_bound) )
    initial_guess.append(np.mean([lower_bound, upper_bound]))


y_min = optimize.minimize(lambda x: synth_ground_truth.predict(x.reshape(1, -1)),
                          x0=initial_guess, bounds=feature_intervals)
print(feature_intervals)
print(y_min)
