import math
import pandas as pd
import sklearn
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

import get_input_data


def split_df(df, split_array):
    # Takes a DataFrame and splits it into 3 sets according to the ratio in the given array
    # split_array must have a length of 3.

    assert(len(split_array == 3))

    split = [int(i / sum(split_array) * len(df)) for i in split_array]

    df_head = df.head(split[0])
    df_mid = df.iloc[(split[0] + 1):(split[0] + split[1])]
    df_tail = df.tail(split[2])

    return [df_head, df_mid, df_tail]


def preprocess_features(df):
    processed_features = pd.DataFrame()
    processed_features["DECHOUR"] = df["TIMESTAMPS"].apply(lambda x: x.dt.hour + x.dt.minute / 1440)  # Turns datetime format into decimalhour, normalised by day
    processed_features["DAYOFWEEK"] = df["TIMESTAMPS"].dt.strftime("%a")     # day of week
    processed_features["MONTHOFYEAR"] = df["TIMESTAMPS"].dt.strftime("%b")     # month of year
    return processed_features


def preprocess_targets(df):
    processed_targets = pd.DataFrame()
    processed_targets["USERID"] = df["USERID"]
    return processed_targets


def construct_feature_columns(numerical_columns_list, catagorical_columns_list, raw_df):

    numerical_features_list = []
    for i in numerical_columns_list:
        current_column = tf.feature_column.numeric_column(key=i)
        numerical_features_list.append(current_column)

    catagorical_features_list = []
    for i in catagorical_columns_list:
        current_column = tf.feature_column.categorical_column_with_vocabulary_list(key=i, vocabulary_list=raw_df[i].unique())
        # current_column = tf.feature_column.indicator_column(catagorical_column=current_column) # May need to wrap within indicator column
        catagorical_features_list.append(current_column)

    feature_column_list = numerical_features_list + catagorical_features_list
    return feature_column_list


def create_input_function(features, targets, shuffle=True, batch_size=1, num_epochs=0):
    input_fn = tf.estimator.inputs.pandas_input_fn(features, y=targets, shuffle=shuffle, batch_size=batch_size, num_epochs=num_epochs)
    return input_fn


def rmse_plot(train, val):
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(train, label="training")
    plt.plot(val, label="validation")
    plt.axis([0, 10, 0, 0.2])  # Lock axis
    plt.legend()
    plt.show()


def train_model(
        train_features,
        train_targets,
        val_features,
        val_targets,
        learning_rate=0.001,
        batch_size=1,
        steps_per_period=50,
        periods=10,
        hidden_units=[1024, 512, 256]
):
    # Create DNN
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)  # Create optimiser - Try variable rate optimisers
    classifier = tf.estimator.DNNClassifier(feature_columns=train_features, hidden_units=hidden_units, optimizer=optimizer)

    # Create input functions
    train_input_fn = create_input_function(train_features, train_targets, batch_size=batch_size, num_epochs=10)
    # Input functions for finding RMSE values
    predict_train_input_fn = create_input_function(train_features, train_targets, shuffle=False, num_epochs=1)
    predict_val_input_fn = create_input_function(val_features, val_targets, shuffle=False, num_epochs=1)

    # Begin Training

    # print statement for RMSE values
    print("  period    | train   | val")
    train_rmse = []
    val_rmse = []

    for period in range(periods):
        # Train Model
        classifier.train(input_fn=train_input_fn, steps=steps_per_period)

        # Compute Predictions
        train_predictions = classifier.predict(input_fn=predict_train_input_fn)
        val_predictions = classifier.predict(input_fn=predict_val_input_fn)

        train_predictions_arr = np.array([item["predictions"][0] for item in train_predictions])
        val_predictions_arr = np.array([item["predictions"][0] for item in val_predictions])

        # Compute Loss
        train_rmse_current_tensor = sklearn.metrics.mean_squared_error(train_targets, train_predictions_arr)
        val_rmse_current_tensor = sklearn.metrics.mean_squared_error(val_targets, val_predictions_arr)

        train_rmse_current = math.sqrt(train_rmse_current_tensor)
        val_rmse_current = math.sqrt(val_rmse_current_tensor)

        # print(period, train_rmse_current, val_rmse_current)
        print("  period %02d : %0.6f, %0.6f" % (period, train_rmse_current, val_rmse_current))

        # Append RMSE to List
        train_rmse.append(train_rmse_current)
        val_rmse.append(val_rmse_current)

    rmse_plot(train_rmse, val_rmse)
    return classifier


def test_model(model, test_features, test_targets):
    # Create test input function
    predict_test_input_fn = create_input_function(test_features, test_targets, shuffle=False, batch_size=1, num_epochs=1)

    # Get predictions as an Array
    test_predictions = model.predict(input_fn=predict_test_input_fn)
    test_predictions = np.array([item["predictions"][0] for item in test_predictions])

    # Use sklearn.metrics to calculate and print RMSE
    test_rmse_current = math.sqrt(metrics.mean_squared_error(test_targets, test_predictions))
    print("Test Data RMSE:", test_rmse_current)


def main():
    raw_df = get_input_data.get_events()    # Get Raw DF

    df_array = split_df(raw_df, [2, 2, 1])  # Split into 3 DFs

    # Assign train, validation and test features + targets
    train_features = preprocess_features(df_array[0])
    train_targets = preprocess_targets(df_array[0])

    val_features = preprocess_features(df_array[1])
    val_targets = preprocess_targets(df_array[1])

    test_features = preprocess_features(df_array[2])
    test_targets = preprocess_targets(df_array[2])

    dnn_classifier = train_model(
        train_features,
        train_targets,
        val_features,
        val_targets,
        learning_rate=0.0005,
        batch_size=1000,
        steps_per_period=100,
        periods=10,
        hidden_units=[1024, 512, 256]
    )

    test_model(dnn_classifier, test_features, test_targets)

main()