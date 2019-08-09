import os
import pickle
import shutil

import tensorflow as tf
from tensorflow import data
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', 15)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', None)


def create_input_function(features, targets, batch_size=1, num_epochs=None, shuffle=False):
    features = {key: np.array(value) for key, value in dict(features).items()}

    ds = data.Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def create_feature_columns(df):

    feature_column_list = []

    feature_column_list.append(tf.feature_column.indicator_column(
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(key="Present", vocabulary_list=['True', 'False'])))

    feature_column_list.append(tf.feature_column.indicator_column(
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(key="Day_of_week", vocabulary_list=df["Day_of_week"].unique())))

    feature_column_list.append(tf.feature_column.indicator_column(
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(key="Day_of_month", vocabulary_list=df["Day_of_month"].unique())))

    feature_column_list.append(tf.feature_column.indicator_column(
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(key="Month_of_year", vocabulary_list=df["Month_of_year"].unique())))

    feature_column_list.append(tf.feature_column.numeric_column(key="Prev_absences"))

    feature_column_list.append(tf.feature_column.numeric_column(key="Users_absent"))

    feature_column_list.append(tf.feature_column.indicator_column(
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(key="USERID", vocabulary_list=df["USERID"].unique())))

    return feature_column_list


def train_model(train_features, train_targets, test_features, test_targets, learning_rate, steps, batch_size,
                model_dir):
    feature_columns = create_feature_columns(train_features)
    label_vocab_list = train_targets['Reason'].unique().tolist()

    optimizer = lambda: tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    #   optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)

    # Create the DNN
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[128, 64],
                                            optimizer=optimizer,
                                            label_vocabulary=label_vocab_list,
                                            n_classes=len(label_vocab_list),
                                            model_dir=model_dir,
                                            config=tf.estimator.RunConfig().replace(save_summary_steps=50))

    # Input functions
    train_fn = lambda: create_input_function(train_features, train_targets, batch_size=batch_size, shuffle=True)

    predict_train_fn = lambda: create_input_function(train_features, train_targets, batch_size=1, num_epochs=1)
    predict_test_fn = lambda: create_input_function(test_features, test_targets, batch_size=1, num_epochs=1)

    periods = 10  # Training periods #TODO set back to 10
    steps_per_period = steps // 10

    print("Training...")

    for period in range(periods):
        classifier.train(input_fn=train_fn, steps=steps_per_period)

        print('Period ' + str(period) + ':')
        print("Training:", classifier.evaluate(predict_train_fn, name="Train"))
        print("Testing:", classifier.evaluate(predict_test_fn, name="Test"))

    print("Training ended.")

    return classifier


# defines and deletes the 'tmp' file if exists (for TensorBoard)
model_dir_path = "tmp/tf"
try:
    if os.path.exists(model_dir_path) and os.path.isdir(model_dir_path):
        shutil.rmtree(model_dir_path)
except FileNotFoundError:
    print('Error while attempting to delete directory ' + model_dir_path)

test_features = pd.read_csv("test_features.csv", index_col=0)
test_targets = pd.read_csv("test_labels.csv", index_col=0)
train_features = pd.read_csv("train_features.csv", index_col=0)
train_targets = pd.read_csv("train_labels.csv", index_col=0)

# Create Classifier
classifier = train_model(train_features=train_features,
                         train_targets=train_targets,
                         test_features=test_features,
                         test_targets=test_targets,
                         learning_rate=0.003,  # 0.003 works
                         steps=10,  # 3000
                         batch_size=1,  # 100 works
                         model_dir=model_dir_path)

# Dump Classifier Object
with open('tensorflow_reason_classifier.pkl', 'wb') as output:
    pickle.dump(classifier, output, pickle.HIGHEST_PROTOCOL)

# Get prediction
predictions = predict_model(classifier, test_features, test_targets)

# Place into DF
predictions_labels_df = pd.DataFrame()
for index, prediction in enumerate(predictions):
    # if index == 25:  # Stop this at 25 items
    #     break
    if index == 0:
        predictions_df = pd.DataFrame(columns=prediction.get("all_classes"))
        predictions_df.columns = [i.decode("utf-8") for i in predictions_df.columns]

    predictions_df.loc[index] = prediction.get("probabilities")
    # predictions_labels_df.loc[index, "Actual Labels"] = prediction.get("classes")[0].decode("utf-8")
    predictions_labels_df.loc[index, "Predicted Labels"] = predictions_df.columns.tolist()[np.argmax(prediction.get("probabilities"))]

predictions_df.to_csv("tensorflow predictions df")
predictions_labels_df.to_csv("tensorflow predictions labels df")

