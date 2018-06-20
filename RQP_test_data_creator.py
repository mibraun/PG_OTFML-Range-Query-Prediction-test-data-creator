import numpy as np
import pandas as pd
import util
from pandas2arff import pandas2arff
from scipy.io import arff
from scipy import optimize
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


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
    X_gt_train = X
    Y_gt_train = np.ravel(Y)

    # MLP regressor with alpha-parameter and hidden layer sizes determined by grid search cross validation
    param_grid = {'classify__alpha': 10.0 ** -np.arange(1, 7),
                      'classify__hidden_layer_sizes': [(i, j, 20, 20) for i in range(20, 21, 1) for j in range(20, 21, 1)]}

    synth_ground_truth_pipe = Pipeline([('normalize', util.FittingNormalizer()), ('classify', MLPRegressor())])
    synth_ground_truth = GridSearchCV(synth_ground_truth_pipe, param_grid)
    synth_ground_truth.fit(X_gt_train, Y_gt_train)
    print("CV score:", synth_ground_truth.best_score_)

    # Original X data with synthetic ground truth Y-values as training data for RQP
    X_RQP_train = X_norm
    Y_RQP_train = np.reshape(synth_ground_truth.predict(X_RQP_train), (N, 1))
    data_train = np.concatenate([X_RQP_train, Y_RQP_train], axis=1)

    # Generate interval-valued test data points for RQP
    # by optimizing on random intervals on synthetic ground truth
    data_test = []
    num_features = X_RQP_train.shape[1]
    for i in range(num_test_data_points):
        if sampling_method == "length_uniform":
            feature_intervals = interval_sample_length(num_features)
        else:
            feature_intervals = interval_sample_uniform(num_features)
        initial_guess = []
        for feature_interval in feature_intervals:
            initial_guess.append(np.mean([feature_interval[0], feature_interval[1]]))

        y_min = optimize.minimize(lambda x: synth_ground_truth.predict(x.reshape(1, -1)),
                                  x0=initial_guess, bounds=feature_intervals)
        y_max = optimize.minimize(lambda x: (-1) * synth_ground_truth.predict(x.reshape(1, -1)),
                                  x0=initial_guess, bounds=feature_intervals)
        feature_intervals.append((synth_ground_truth.predict(y_min.x.reshape(1, -1))[0],
                                  synth_ground_truth.predict(y_max.x.reshape(1, -1))[0]))
        data_test.append(feature_intervals)

    # unpack intervals for test data
    data_test_unpacked = []
    for data in data_test:
        feature_intervals_unpacked = []
        for feature in data:
            feature_intervals_unpacked.append(feature[0])
            feature_intervals_unpacked.append(feature[1])
        data_test_unpacked.append(feature_intervals_unpacked)

    # Put all data in panda data frames with nicely named columns
    data_train_df = pd.DataFrame(data_train)
    data_test_df = pd.DataFrame(data_test_unpacked)
    column_names_train = {}
    column_names_test = {}
    for feature in range(num_features):
        column_names_train[feature] = 'x' + str(feature)
        column_names_test[feature * 2] = 'x' + str(feature) + '_lower'
        column_names_test[feature * 2 + 1] = 'x' + str(feature) + '_upper'
    column_names_train[num_features] = 'y'
    column_names_test[num_features * 2] = 'y_min'
    column_names_test[num_features * 2 + 1] = 'y_max'
    data_train_df.rename(columns=column_names_train, inplace=True)
    data_test_df.rename(columns=column_names_test, inplace=True)

    return data_train_df, data_test_df, synth_ground_truth


if __name__ == "__main__":
    file_path = "regression/cpu.small.arff"
    data_train_df, data_test_df, sgt_model = create_test_data("regression/cpu.small.arff")
    pandas2arff(data_test_df, file_path + "_RQPtest.arff", wekaname=file_path + "_RQP test data")
    pandas2arff(data_train_df, file_path + "_RQPtrain.arff", wekaname=file_path + "_RQP training data")
    #with open(file_path + "_RQP_sgt", 'w') as sgt_file:
    #    sgt_file.write(str(sgt_model))
