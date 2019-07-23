import math
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import data
import os
import shutil


def pp_feature(df):
    feature_df = pd.DataFrame()
    # Take the following features of the DF: USERID, Day, Present
    feature_df['USERID'] = df['USERID']
    feature_df['Present'] = df['Present']

    # For 'Day', convert to Give it synthetic features: Day_of_week (0 = Monday, 6 = Sunday) // Day_of_month // Month_of_year
    feature_df['Day_of_week'] = [day.weekday() for day in df['Day']]
    feature_df['Day_of_month'] = [day.day for day in df['Day']]
    feature_df['Month_of_year'] = [day.month for day in df['Day']]

    return feature_df


def pp_targets(df):
    # Take the targets of the DF: Reason
    target_df = df[['Reason']]
    return target_df


def create_input_function(features, targets, batch_size=1, num_epochs=None, shuffle=False):
    features = {key: np.array(value) for key, value in dict(features).items()}

    ds = data.Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


# Hard-code feature columns for now
def create_feature_columns(df):

    feature_column_list = []

    feature_column_list.append(tf.feature_column.indicator_column(
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(key="USERID", vocabulary_list=df["USERID"].unique())))

    feature_column_list.append(tf.feature_column.indicator_column(
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(key="Present", vocabulary_list=['True', 'False'])))

    feature_column_list.append(tf.feature_column.indicator_column(
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(key="Day_of_week", vocabulary_list=df["Day_of_week"].unique())))

    feature_column_list.append(tf.feature_column.indicator_column(
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(key="Day_of_month", vocabulary_list=df["Day_of_month"].unique())))

    feature_column_list.append(tf.feature_column.indicator_column(
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(key="Month_of_year", vocabulary_list=df["Month_of_year"].unique())))

    return feature_column_list


def train_model(train_features, train_targets, test_features, test_targets, learning_rate, steps, batch_size, hidden_units, model_dir):
    feature_columns = create_feature_columns(train_features)
    label_vocab_list = train_targets['Reason'].unique()
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    # Create the DNN
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns, 
                                            hidden_units=hidden_units,
                                            optimizer=optimizer,
                                            label_vocabulary=label_vocab_list,
                                            num_classes=len(label_vocab_list), 
                                            model_dir=model_dir,
                                            config=tf.estimator.RunConfig().replace(save_summary_steps=10))
    
    # Input functions
    train_fn = lambda: create_input_function(train_features, train_targets, batch_size=batch_size, shuffle=True)
    
    predict_train_fn = lambda: create_input_function(train_features, train_targets, batch_size=1, num_epochs=1)
    predict_test_fn = lambda: create_input_function(test_features, test_targets, batch_size=1, num_epochs=1)

    periods = 10  # Training periods
    steps_per_period = steps // 10

    for period in range(periods):
        classifier.train(input_fn=train_fn, steps=steps)

        classifier.evaluate(predict_train_fn, name="Train")
        classifier.evaluate(predict_test_fn, name="Test")

    return classifier


def predict_model(classifier, features, targets):
    predict_input_fn = create_input_function(features, targets, batch_size=1, num_epochs=1)

    predict_results = classifier.predict(predict_input_fn, predict_keys="probabilities")

    print(predict_results.get("probabilities"))


def main():
    # defines and deletes the 'tmp' file if exists (for TensorBoard)
    model_dir_path = "tmp/tf"
    try:
        if os.path.exists(model_dir_path) and os.path.isdir(model_dir_path):
            shutil.rmtree(model_dir_path)
    except FileNotFoundError:
        print('Error while attempting to delete directory ' + model_dir_path)

    raw_df = pd.read_csv("reason_df.csv")
    raw_df['Day'] = pd.to_datetime(raw_df['Day'])
    raw_df = raw_df.reindex(np.random.permutation(raw_df.index))
    raw_df.reset_index(inplace=True)

    df_train = raw_df.head(math.floor(len(raw_df) * 0.7))
    df_test = raw_df.tail(math.floor(len(raw_df) * 0.3))

    train_features = pp_feature(df_train)
    test_features = pp_feature(df_test)
    
    train_targets = pp_targets(df_train)
    test_targets = pp_targets(df_test)

    classifier = train_model(train_features=train_features,
                             train_targets=train_targets,
                             test_features=test_features,
                             test_targets=test_targets,
                             learning_rate=0.0003,
                             steps=1000,
                             batch_size=1,
                             hidden_units=[5, 1],
                             model_dir=model_dir_path)

    predict_model(classifier, test_targets, test_targets)


main()
