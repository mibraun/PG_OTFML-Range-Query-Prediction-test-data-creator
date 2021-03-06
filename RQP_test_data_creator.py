import numpy as np
import pandas as pd
import os
from pandas2arff import pandas2arff
from sklearn.preprocessing import StandardScaler
from scipy.io import arff
from scipy import optimize
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def create_test_data(file_path, num_test_data_points=None):
    """
    Generates semi-synthetic data consisting of a set of precise training instances
    and a set of interval-valued test instances based on a given data set.
    Currently only supports numeric features.

    :param file_path:  Path of the data set to base the semi-synthetic data on. Should be in .arff format.
    :param num_test_data_points:  The number of interval-valued test instances to create. Default: 30% of training data size.
    """
    raw_data = pd.DataFrame(arff.loadarff(file_path)[0])
    X = raw_data.iloc[:, :-1].get_values()
    Y = raw_data.iloc[:, -1:]
    N = raw_data.shape[0]
    if num_test_data_points is None:
        num_test_data_points = int(np.floor(0.3 * N))

    create_test_data_internal(N=N, X=X, Y=Y, dataset_name=file_path, num_test_data_points=num_test_data_points,
                              numeric_features=range(len(X[0])))


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


# Creates the semi-synthetic test data & training data based on a given dataset
def create_test_data_internal(X, Y, N, numeric_features, dataset_name, num_test_data_points=None, noises=[0, 0.1, 0.3, 0.5]):
    if num_test_data_points is None:
        num_test_data_points = N

    X_gt_train, Y_gt_train = standardize_data(X=X, Y=Y)

    synth_ground_truth = train_neural_network(X_gt_train=X_gt_train, Y_gt_train=Y_gt_train)

    # Original X data with synthetic ground truth Y-values as training data for RQP
    X_RQP_train = X_gt_train
    Y_RQP_train = np.reshape(synth_ground_truth.predict(X_RQP_train), (N, 1))

    column_names_train = {}

    num_features = X_RQP_train.shape[1]

    for feature in range(num_features):
        column_names_train[feature] = 'x' + str(feature)
    column_names_train[num_features] = 'y'

    train_data = []
    for noise in noises:
        y_copy = np.copy(Y_RQP_train)
        for i in range(len(y_copy)):
            y_copy[i] = y_copy[i] + np.random.normal(0, noise)
        data_train = np.concatenate([X_RQP_train, y_copy], axis=1)
        data_train_df = pd.DataFrame(data_train)
        data_train_df.rename(columns=column_names_train, inplace=True)
        train_data.append(data_train_df)

    data_test_df = create_rqp_data(X_RQP_train=X_RQP_train,
                                   numeric_features=numeric_features,
                                   synth_ground_truth=synth_ground_truth,
                                   num_test_data_points=num_test_data_points)

    write_to_file(data_train_dfs=train_data, data_test_df=data_test_df, noises=noises, dataset_name=dataset_name)


def standardize_data(X, Y):
    print("Standardizing...")
    standardized_y = StandardScaler().fit_transform(Y)
    X_norm = StandardScaler().fit_transform(X)
    return X_norm, np.ravel(standardized_y)


def train_neural_network(X_gt_train, Y_gt_train):
    print("Optimizing DNN...")

    # MLP regressor with alpha-parameter and hidden layer sizes determined by grid search cross validation
    param_grid = {'classify__alpha': 10.0 ** -np.arange(1, 7),
                  'classify__activation': ['logistic'],
                  'classify__hidden_layer_sizes': [(layer1, layer2)
                                                   for layer1 in range(10, 28, 4)
                                                   for layer2 in range(10, 28, 4)]}

    synth_ground_truth_pipe = Pipeline([('classify', MLPRegressor())])
    synth_ground_truth = GridSearchCV(synth_ground_truth_pipe, param_grid, n_jobs=1)
    synth_ground_truth.fit(X_gt_train, Y_gt_train)
    print("CV score:", synth_ground_truth.best_score_)
    print("Finished optimizing, params are:")
    print(synth_ground_truth.best_params_)
    return synth_ground_truth


def create_rqp_data(X_RQP_train, synth_ground_truth, numeric_features, num_test_data_points):
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
        feature_intervals = interval_sample_random_features(X_RQP_train)

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
            y_mins.append(synth_ground_truth.predict(y_min_result.x.reshape(1, -1))[0])
            y_maxs.append(synth_ground_truth.predict(y_max_result.x.reshape(1, -1))[0])
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
    data_test_df = pd.DataFrame(data_test_unpacked)
    column_names_test = {}
    for feature in range(num_features):
        column_names_test[feature * 2] = 'x' + str(feature) + '_lower'
        column_names_test[feature * 2 + 1] = 'x' + str(feature) + '_upper'
    column_names_test[num_features * 2] = 'y_min'
    column_names_test[num_features * 2 + 1] = 'y_max'
    data_test_df.rename(columns=column_names_test, inplace=True)
    return data_test_df


# Samples intervals by choosing two values x_i and x'_i for each feature i from the data and using them as interval endpoints.
def interval_sample_random_features(X):
    feature_intervals = []
    for feature in range(X.shape[1]):
        feature_column = X[:, feature]
        random_sample_one = np.random.choice(a=feature_column)
        random_sample_two = np.random.choice(a=feature_column)
        feature_intervals.append((np.min([random_sample_one, random_sample_two]), np.max([random_sample_one, random_sample_two])))
    return feature_intervals


def write_to_file(data_train_dfs, data_test_df, noises, dataset_name, sampling_method="random_features"):
    stripped_dataset_name = dataset_name.replace('.arff', '')
    stripped_dataset_name = stripped_dataset_name.replace('regression/', '')
    pandas2arff(data_test_df,
                "data/" + dataset_name + "/" + sampling_method + "/" + stripped_dataset_name + "_RQPtest.arff",
                wekaname=file_path + "_RQP_test_data")

    for i in range(len(noises)):
        pandas2arff(data_train_dfs[i],
                    "data/" + dataset_name + "/" + sampling_method + "/" + stripped_dataset_name + "_noise_" + str(
                        noises[i]) + "_RQPtrain.arff",
                    wekaname=file_path + "_RQP_training_data")


if __name__ == "__main__":
    sampling_method = "random_features"
    file_paths = ["regression/boston.arff", "regression/bank32nh.arff", "regression/bank8FM.arff",
                  "regression/bodyfat.arff", "regression/cpu.small.arff", "regression/cal.housing.arff",
                  "regression/elevators.arff", "regression/house8L.arff", "regression/kin8nm.arff",
                  "regression/machine.cpu.arff"
                  ]
    for file_path in file_paths:
        try:
            os.makedirs("data/" + file_path + "/" + sampling_method, exist_ok=True)
            create_test_data(file_path)
        except RuntimeError as err:
            print(err)
            pass
