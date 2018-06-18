import numpy as np
import pandas as pd
from pandas2arff import pandas2arff
from scipy.io import arff
from scipy import optimize
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV

# Samples uniformly from all intervals on [0,1)
def interval_sample_uniform(num_features):
    feature_intervals = []
    for feature in range(num_features):
        x1 = np.random.random_sample()
        x2 = np.random.random_sample()
        lower_bound = np.min([x1, x2])
        upper_bound = np.max([x1, x2])
        feature_intervals.append(
            (lower_bound, upper_bound)
        )
    return feature_intervals


# Samples intervals on [0,1), such that all interval lengths are represented uniformly
def interval_sample_length(num_features):
    feature_intervals = []
    for feature in range(num_features):
        interval_length = np.random.random_sample()
        lower_bound = interval_length * np.random.random_sample()
        upper_bound = lower_bound + interval_length
        feature_intervals.append(
            (lower_bound, upper_bound)
        )
    return feature_intervals


def create_test_data(file_path, sampling_method="uniform", num_test_data_points=None):
    raw_data = pd.DataFrame(arff.loadarff(file_path)[0])
    X = raw_data.iloc[:, :-1]
    Y = raw_data.iloc[:, -1:]
    X_norm = normalize(X, axis=0)
    N = raw_data.shape[0]
    if num_test_data_points == None:
        num_test_data_points = N
    # Training data for synthetic ground truth
    X_gt_train = X_norm
    Y_gt_train = np.ravel(Y)

    # MLP regressor with alpha-parameter and hidden layer sizes determined by grid search cross validation
    mlp_param_grid = {'alpha': 10.0 ** -np.arange(1, 3),
                      'hidden_layer_sizes': [(i, j) for i in range(10, 21) for j in range(10, 21)]}
    synth_ground_truth = GridSearchCV(MLPRegressor(), mlp_param_grid)

    synth_ground_truth.fit(X_gt_train, Y_gt_train)

    # Original X data with synthetic ground truth Y-values as training data for RQP
    X_RQP_train = X_norm
    Y_RQP_train = np.reshape(synth_ground_truth.predict(X_RQP_train), (N, 1))
    data_train = np.concatenate([X_RQP_train, Y_RQP_train], axis=1)
    #pandas2arff(pd.DataFrame(data_train), "cpu.small_RQP_train.arff", wekaname="test")

    print("CV score:", synth_ground_truth.best_score_)

    # Generate interval-valued test data points for RQP
    # by optimizing on random intervals on synthetic ground truth
    data_test = []
    num_features = X_RQP_train.shape[1]
    #for i in range(num_test_data_points):
    if sampling_method == "length_uniform":
        feature_intervals = interval_sample_length(num_features)
    else:
        feature_intervals = interval_sample_uniform(num_features)
    initial_guess = []
    for feature_interval in feature_intervals:
        initial_guess.append(np.mean([feature_interval[0], feature_interval[1]]))

    y_min = optimize.minimize(lambda x: synth_ground_truth.predict(x.reshape(1, -1)),
                              x0=initial_guess, bounds=feature_intervals)
    print(feature_intervals)
    print("Success: ", y_min.success, " min: ", synth_ground_truth.predict(y_min.x.reshape(1, -1)))

    return pd.DataFrame(data_train), pd.DataFrame(data_test), synth_ground_truth


if __name__ == "__main__":
    create_test_data("regression/cpu.small.arff")
