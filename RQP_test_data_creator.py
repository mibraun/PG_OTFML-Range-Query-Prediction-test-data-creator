import numpy as np
import pandas as pd
import os
import util
import random
import pickle
from pandas2arff import pandas2arff
from CustomScaler import CustomScaler
from sklearn.preprocessing import StandardScaler
from scipy.io import arff
from scipy import optimize
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from random import randint

### Constants:

## The average depth of a tree in the heuristic search algorithm
k = 8


# Samples uniformly from all intervals on [0,1)
def interval_sample_uniform(num_features, ignore=None):
    feature_intervals = []
    for feature in range(num_features):
        if feature in ignore:
            continue
        x1 = np.random.random_sample()
        x2 = np.random.random_sample()
        lower_bound = np.min([x1, x2])
        upper_bound = np.max([x1, x2])
        feature_intervals.append(
            (lower_bound, upper_bound)
        )

    return feature_intervals


def interval_sample_length_exponential(numeric_min, numeric_max, categoric_features):
    feature_intervals = []
    ## categoric features
    # we assume one hot encoding
    if len(categoric_features) is not 0:
        categoric_max_interval = len(categoric_features)
        print("Categoric max interval length: " + str(categoric_max_interval))
        categoric_expected_interval_length = (float(k) * categoric_max_interval) / ((2 ** k) - 1)
        print("Categoric expected interval length: " + str(categoric_expected_interval_length))
        categoric_random_interval_length = np.random.exponential(scale=0.5)
        print("Categoric actual interval length: " + str(categoric_random_interval_length))
        indices = random.sample(range(0, len(categoric_features)), int(np.ceil(categoric_random_interval_length)))

        for i in range(0, len(categoric_features)):
            if i in indices:
                feature_intervals.append((1, 1))
            else:
                feature_intervals.append((0, 0))

    print("Numeric max interval length: " + str(numeric_max))
    ## numeric features
    for i in range(0, len(numeric_min)):
        numeric_length = numeric_max[i] - numeric_min[i]
        numeric_expected_interval_length = (float(k) * numeric_length) / ((2 ** k) - 1)
        numeric_random_length = np.random.exponential(scale=numeric_expected_interval_length)
        max_lower_bound = numeric_max[i] - numeric_random_length
        lower_bound = np.random.uniform(low=numeric_min[i], high=max_lower_bound)
        feature_intervals.append((lower_bound, lower_bound + numeric_random_length))
        print("Numeric expected interval length: " + str(numeric_expected_interval_length))
        print("Numeric actual interval length: " + str(numeric_random_length))
    return feature_intervals


# Samples intervals on [0,1), such that all interval lengths are represented uniformly
def interval_sample_length(numeric_features, categoric_features):
    feature_intervals = []

    for feature in range(0, len(numeric_features)):
        interval_length = np.random.random_sample()
        lower_bound = interval_length * np.random.random_sample()
        upper_bound = lower_bound + interval_length
        feature_intervals.append(
            (lower_bound, upper_bound)
        )

    for i in range(0, len(categoric_features)):
        value = randint(0, 1)
        feature_intervals.append((value, value))
    return feature_intervals


def interval_sample_random_points(numeric_features, categoric_features, X):
    feature_intervals = []
    randindex1 = randint(0, len(X) - 1)
    randindex2 = randint(0, len(X) - 1)
    random_point1 = X[randindex1]
    random_point2 = X[randindex2]

    for feature in numeric_features:
        lower_bound = min(random_point1[feature], random_point2[feature])
        upper_bound = max(random_point1[feature], random_point2[feature])
        feature_intervals.append(
            (lower_bound, upper_bound)
        )

    for feature in categoric_features:
        lower_bound = max(random_point1[feature], random_point2[feature])
        upper_bound = max(random_point1[feature], random_point2[feature])
        feature_intervals.append(
            (lower_bound, upper_bound)
        )

    return feature_intervals


def stadardize_data(categoric_features, X, Y, numeric_features):
    print("Standardizing...")

    # We don't want to standardize the categorical data
    if len(categoric_features) is 0:
        X_norm = StandardScaler().fit_transform(X)
    else:
        X_norm = CustomScaler(bin_vars_index=categoric_features, cont_vars_index=numeric_features).fit_transform(X)
    return X_norm, np.ravel(Y)


def train_neural_network(X_gt_train, Y_gt_train):
    print("Optimizing DNN...")

    # MLP regressor with alpha-parameter and hidden layer sizes determined by grid search cross validation
    param_grid = {'classify__alpha': 10.0 ** -np.arange(1, 4),
                  'classify__hidden_layer_sizes': [(layer1, layer2, layer3, layer4)
                                                   for layer1 in range(10, 28, 4)
                                                   for layer2 in range(10, 28, 4)
                                                   for layer3 in range(10, 28, 4)
                                                   for layer4 in range(10, 28, 4)]}

    synth_ground_truth_pipe = Pipeline([('classify', MLPRegressor())])
    synth_ground_truth = GridSearchCV(synth_ground_truth_pipe, param_grid, n_jobs=1)
    synth_ground_truth.fit(X_gt_train, Y_gt_train)
    print("CV score:", synth_ground_truth.best_score_)
    print("Finished optimizing, params are:")
    print(synth_ground_truth.best_params_)
    return synth_ground_truth


def create_rqp_data(X_RQP_train, synth_ground_truth, numeric_features, num_test_data_points, categoric_features,
                    data_train, sampling_method, sigma):
    # Generate interval-valued test data points for RQP
    # by optimizing on random intervals on synthetic ground truth
    data_test = []
    num_features = X_RQP_train.shape[1]

    numeric_min = []
    numeric_max = []
    for feature in numeric_features:
        feature_column = X_RQP_train[:, feature]
        numeric_min.append(np.min(feature_column))
        numeric_max.append(np.max(feature_column))

    for i in range(num_test_data_points):
        NUM_RANDOM_RESTARTS = 10
        if sampling_method == "length_uniform":
            feature_intervals = interval_sample_length(num_features, categoric_features)
        elif sampling_method == "uniform":
            feature_intervals = interval_sample_uniform(numeric_features=numeric_features,
                                                        categoric_features=categoric_features)
        elif sampling_method == "random_points":
            feature_intervals = interval_sample_random_points(numeric_features=numeric_features,
                                                              categoric_features=categoric_features, X=X_RQP_train)
        elif sampling_method == "length_exponential":
            feature_intervals = interval_sample_length_exponential(numeric_min=numeric_min, numeric_max=numeric_max,
                                                                   categoric_features=categoric_features)

        # Restarts for blackbox optimization: middle of intervals plus 10 randomly chosen restarts
        initial_guesses = []
        mid_guess = []
        for feature_interval in feature_intervals:
            mid_guess.append(np.mean([feature_interval[0], feature_interval[1]]))
        initial_guesses.append(mid_guess)
        for j in range(NUM_RANDOM_RESTARTS):
            rand_guess = []
            for feature_interval in feature_intervals:
                rand_feature_value = (feature_interval[1] - feature_interval[0]) \
                                     * np.random.random_sample() + feature_interval[0]
                rand_guess.append(rand_feature_value)
            initial_guesses.append(rand_guess)
        y_mins = []
        y_maxs = []
        for initial_guess in initial_guesses:
            y_min_result = optimize.minimize(lambda x: synth_ground_truth.predict(x.reshape(1, -1)),
                                             x0=initial_guess, bounds=feature_intervals)
            y_max_result = optimize.minimize(lambda x: (-1) * synth_ground_truth.predict(x.reshape(1, -1)),
                                             x0=initial_guess, bounds=feature_intervals)
            y_mins.append(synth_ground_truth.predict(y_min_result.x.reshape(1, -1))[0] + np.random.normal(0, sigma))
            y_maxs.append(synth_ground_truth.predict(y_max_result.x.reshape(1, -1))[0] + np.random.normal(0, sigma))
        y_min = np.min(y_mins)
        y_max = np.max(y_maxs)
        feature_intervals.append((y_min, y_max))
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
    return data_train_df, data_test_df


def write_to_file(data_train_df, data_test_df, noise, sampling_method, dataset_name):
    stripped_dataset_name = dataset_name.replace('.arff', '')
    stripped_dataset_name = stripped_dataset_name.replace('regression/', '')
    pandas2arff(data_test_df,
                "data/" + dataset_name + "/" + sampling_method + "/" + stripped_dataset_name + "_noise_" + str(
                    noise) + "_RQPtest.arff",
                wekaname=file_path + "_RQP_test_data")
    pandas2arff(data_train_df,
                "data/" + dataset_name + "/" + sampling_method + "/" + stripped_dataset_name + "_noise_" + str(
                    noise) + "_RQPtrain.arff",
                wekaname=file_path + "_RQP_training_data")


# Creates the test-data & training data based on the given
def create_test_data_internal(X, Y, N, numeric_features, dataset_name, categoric_features=[], num_test_data_points=None,
                              sampling_method="random_points", noises=[0.1, 0.01, 0.001]):
    if num_test_data_points is None:
        num_test_data_points = N

    X_gt_train, Y_gt_train = stadardize_data(categoric_features=categoric_features, X=X, Y=Y,
                                             numeric_features=numeric_features)

    synth_ground_truth = train_neural_network(X_gt_train=X_gt_train, Y_gt_train=Y_gt_train)

    # Original X data with synthetic ground truth Y-values as training data for RQP
    X_RQP_train = X_gt_train
    Y_RQP_train = np.reshape(synth_ground_truth.predict(X_RQP_train), (N, 1))
    data_train = np.concatenate([X_RQP_train, Y_RQP_train], axis=1)
    for noise in noises:
        data_train_df, data_test_df = create_rqp_data(X_RQP_train=X_RQP_train, data_train=data_train,
                                                      numeric_features=numeric_features,
                                                      categoric_features=categoric_features, sigma=noise,
                                                      synth_ground_truth=synth_ground_truth,
                                                      num_test_data_points=num_test_data_points,
                                                      sampling_method=sampling_method)

        write_to_file(data_train_df=data_train_df, data_test_df=data_test_df, dataset_name=dataset_name, noise=noise,
                      sampling_method=sampling_method)


def create_test_data(file_path, sampling_method="uniform", num_test_data_points=None):
    raw_data = pd.DataFrame(arff.loadarff(file_path)[0])
    X = raw_data.iloc[:, :-1].get_values()
    Y = raw_data.iloc[:, -1:]
    N = raw_data.shape[0]
    return create_test_data_internal(N=N, X=X, Y=Y, dataset_name=file_path, sampling_method=sampling_method,
                                     num_test_data_points=num_test_data_points, numeric_features=range(len(X[0])),
                                     categoric_features=[])


def read_data_from_arff(train_file, test_file):
    """Reads training and test data for range query prediction from arff files
    and returns them as five numpy arrays.

    Arguments:
        train_file -- file path of the training data. Should be in .arff format.
        test_file -- file path of the test data. Should be in .arff format.

    Returns:
        X_train -- 2-dimensional numpy array containing the training data features
        Y_train -- 1-dimensional numpy array containing the training data target
        X_test -- 2-dimensional numpy array containing the range query test data features in the form
                    x1_min, x1_max, x2_min, x2_max, ...
        Y_test_min -- 1-dimensional numpy array containing the test data target minima
        Y_test_max -- 1-dimensional numpy array containing the test data target maxima
    """

    raw_data_train = pd.DataFrame(arff.loadarff(train_file)[0])
    raw_data_test = pd.DataFrame(arff.loadarff(test_file)[0])
    X_train = raw_data_train.iloc[:, :-1].get_values()
    Y_train = raw_data_train.iloc[:, -1:]
    X_test = raw_data_test.iloc[:, :-2].get_values()
    Y_test_min = raw_data_test.iloc[:, -2:-1]
    Y_test_max = raw_data_test.iloc[:, -1:]
    return X_train, np.ravel(Y_train), X_test, np.ravel(Y_test_min), np.ravel(Y_test_max)


if __name__ == "__main__":
    sampling_method = "length_exponential"
    file_paths = ["regression/bodyfat.arff","regression/pollution.arff"
                  ]
    for file_path in file_paths:
        try:
            os.makedirs("data/" + file_path + "/" + sampling_method, exist_ok=True)
            create_test_data(file_path, sampling_method=sampling_method, num_test_data_points=150)
        except RuntimeError as err:
            print(err)
            pass
