import math
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

import get_input_data


def split_df(df, split_array):
    # Takes a DataFrame and splits it into 3 sets according to the ratio in the given array
    # split_array must have a length of 3.

    assert(len(split_array) == 3)

    split = [int(i / sum(split_array) * len(df)) for i in split_array]

    df_head = df.head(split[0])
    df_mid = df.iloc[(split[0] + 1):(split[0] + split[1])]
    df_tail = df.tail(split[2])

    return [df_head, df_mid, df_tail]


def get_decimal_hour(events):
    decimal_hour = (events.dt.hour + events.dt.minute / 60)
    return decimal_hour


def preprocess_features(df):
    processed_features = pd.DataFrame()
    processed_features["DECHOUR"] = get_decimal_hour(df["TIMESTAMPS"]).apply(lambda x: x / 24)  # Turns datetime format into decimalhour, normalised by day
    processed_features["DAYOFWEEK"] = df["TIMESTAMPS"].dt.strftime("%w")     # day of week
    processed_features["MONTHOFYEAR"] = df["TIMESTAMPS"].dt.strftime("%-m")     # month of year
    processed_features["TERMINALSN"] = df["TERMINALSN"]
    processed_features["EVENTID"] = df["EVENTID"]

    # # debugging:
    # print(type(processed_features["DECHOUR"][0]))
    # print(type(processed_features["DAYOFWEEK"][0]))
    # print(type(processed_features["MONTHOFYEAR"][0]))
    # print(type(processed_features["TERMINALSN"][0]))
    # print(type(processed_features["EVENTID"][0]))
    #
    # print('processed features:\n', processed_features)

    return processed_features


def preprocess_targets(df):
    processed_targets = pd.DataFrame()
    processed_targets["USERID"] = df["USERID"].apply(lambda x: str(x))
    return processed_targets


def construct_feature_columns(numerical_columns_list, catagorical_columns_list, vocab_df):
    numerical_features_list = []
    for i in numerical_columns_list:
        current_column = tf.feature_column.numeric_column(key=i)
        numerical_features_list.append(current_column)

    categorical_features_list = []
    for i in catagorical_columns_list:
        current_column = tf.feature_column.categorical_column_with_vocabulary_list(key=i, vocabulary_list=vocab_df[i].unique())
        current_column = tf.feature_column.indicator_column(categorical_column=current_column) # May need to wrap within indicator column
        categorical_features_list.append(current_column)

    feature_column_list = numerical_features_list + categorical_features_list
    return feature_column_list


def create_input_function(features, targets, shuffle=True, batch_size=1, num_epochs=None):

    # APPROACH 3: Directly turning it into a dict-list tuple
    # turn features DataFrame into Dict - input feature is a key, and then a list of values for the training batch
    feature_dict = dict()

    for i in features.columns:
        feature_dict[str(i)] = features[i].tolist()

    # turn targets DataFrame into a List - these are our labels
    label_list = targets[targets.columns[0]].tolist()
    return feature_dict, label_list


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
    numerical_features = ["DECHOUR"]
    categorical_features = ["DAYOFWEEK", "MONTHOFYEAR", "TERMINALSN", "EVENTID"]

    # Create a vocab DataFrame by concatenating the given DFs.
    # NOTE: Should add test_features and test_targets to this later on as well.
    features_vocab_df = train_features.append(val_features)
    feature_columns = construct_feature_columns(numerical_features, categorical_features, features_vocab_df)

    # Prepare label_vocab
    label_vocab_list = train_targets["USERID"].unique()
    label_vocab_list = label_vocab_list.tolist()
    label_vocab_list = [str(i) for i in label_vocab_list]

    # Create DNN
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)  # Create optimiser - Try variable rate optimisers
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=hidden_units, optimizer=optimizer, label_vocabulary=label_vocab_list, n_classes=len(label_vocab_list))

    # Create input functions
    train_input_fn = lambda: create_input_function(train_features, train_targets, batch_size=batch_size, num_epochs=10)

    # Input functions for finding RMSE values
    predict_val_input_fn = lambda: create_input_function(val_features, val_targets, shuffle=False, num_epochs=1)

    # ----- Begin Training -----

    # ignore periods for now
    total_steps = steps_per_period * periods
    classifier.train(input_fn=train_input_fn, steps=total_steps)
    print("classifier gay")

    evaluate_model(classifier, train_features, train_targets)

    return classifier


# Function that tests a model against a set of features and targets;
# Verbose: Checks and prints the result of every single one
def evaluate_model(model, features, targets, verbose=False, name=None):

    print("Evaluating...")

    evaluate_result = model.evaluate(
        input_fn=lambda: create_input_function(features, targets, shuffle=False, num_epochs=1, batch_size=1),
        name=name)

    print("Evaluation results: " + name)

    print(evaluate_result)

    for r_key in evaluate_result:
        if verbose:
            print("  {}, was: {}".format(r_key, evaluate_result[r_key]))


def main():
    raw_df = get_input_data.get_events()  # Get Raw DF
    # raw_df = get_input_data.get_events_from_csv("SECAMS_common_user_id.csv")

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

    # test_model(dnn_classifier, test_features, test_targets)


main()