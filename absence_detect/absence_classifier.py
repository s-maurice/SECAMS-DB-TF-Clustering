import math
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import data
import os
import shutil


def pp_feature(df):

    def one_hot_encode(boolean):
        if boolean:
            return "1"
        else:
            return "0"

    feature_df = pd.DataFrame()
    # Take the following features of the DF: USERID, Day, Present
    # feature_df['USERID'] = [str(i) for i in df['USERID']]
    feature_df['Present'] = [one_hot_encode(i) for i in df['Present']]

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

    # feature_column_list.append(tf.feature_column.indicator_column(
    #     categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(key="USERID", vocabulary_list=df["USERID"].unique())))

    feature_column_list.append(tf.feature_column.indicator_column(
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(key="Present", vocabulary_list=['True', 'False'])))

    feature_column_list.append(tf.feature_column.indicator_column(
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(key="Day_of_week", vocabulary_list=df["Day_of_week"].unique())))

    feature_column_list.append(tf.feature_column.indicator_column(
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(key="Day_of_month", vocabulary_list=df["Day_of_month"].unique())))

    feature_column_list.append(tf.feature_column.indicator_column(
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(key="Month_of_year", vocabulary_list=df["Month_of_year"].unique())))

    return feature_column_list


def train_model(train_features, train_targets, test_features, test_targets, learning_rate, steps, batch_size, model_dir):
    feature_columns = create_feature_columns(train_features)
    label_vocab_list = train_targets['Reason'].unique().tolist()

    optimizer = lambda: tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
#   optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)

    # Create the DNN
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[128, 64, 16],
                                               optimizer=optimizer,
                                               label_vocabulary=label_vocab_list,
                                               n_classes=len(label_vocab_list),
                                               model_dir=model_dir,
                                               config=tf.estimator.RunConfig().replace(save_summary_steps=50))
    
    # Input functions
    train_fn = lambda: create_input_function(train_features, train_targets, batch_size=batch_size, shuffle=True)
    
    predict_train_fn = lambda: create_input_function(train_features, train_targets, batch_size=1, num_epochs=1)
    predict_test_fn = lambda: create_input_function(test_features, test_targets, batch_size=1, num_epochs=1)

    periods = 10  # Training periods
    steps_per_period = steps // 10

    print("Training...")

    for period in range(periods):
        classifier.train(input_fn=train_fn, steps=steps_per_period)

        print('Period ' + str(period) + ':')
        print("Training:", classifier.evaluate(predict_train_fn, name="Train"))
        print("Testing:", classifier.evaluate(predict_test_fn, name="Test"))

    print("Training ended.")

    return classifier


def predict_model(classifier, features, targets):
    # A method that takes in a model, and returns a generator that yields predictions
    # Note: Each iteration through predict_results returns a dict with the following keys:
    #  logits
    #  probabilities    (use this)
    #  class_ids
    #  classes
    #  all_class_ids
    #  all_classes      (use this too)
    predict_input_fn = lambda: create_input_function(features, targets, batch_size=1, num_epochs=1)

    predict_results = classifier.predict(predict_input_fn)

    i = 0
    for idx, prediction in enumerate(predict_results):
        if i == 10:
            break
        for key, value in prediction.items():
            print(key, ":", value)
        i += 1

    return predict_results


def test_result_plotter(predict_results, num_results):
    # A method that takes in a generator of predictions, and plots the first few (num_results) onto a PLT figure.
    # --- Getting a Dict from predict_results (as label -> list of probabilities)
    result_dict = {}

    # Iterate over predict_results and enter the first predictions in to the dataframe
    for idx, prediction in enumerate(predict_results):
        # break if the index is equal to number
        if idx == num_results:
            break

        # Get the list of probabilities, list of classes, and actual class
        probabilities = prediction.get("probabilities")
        classes = [label.decode() for label in prediction.get("all_classes")]
        actual_class = prediction.get("class_ids")[0]  # [0] because .get("classes") returns an array of 1 item

        # Enumerate through classes; add and update the key-value pairs
        for i, label in enumerate(classes):
            # If label exists, update the list
            if label in result_dict.keys():
                result_dict[label].append(probabilities[i])
            else:
                result_dict[label] = [probabilities[i]]

        # Add/update an additional key-value pair: Actual class
        if "actual_class" in result_dict.keys():
            result_dict["actual_class"].append(actual_class)
        else:
            result_dict["actual_class"] = [actual_class]

    print(result_dict)

    # --- Plotting result_dict
    # Make a figure with axes
    fig, axes = plt.subplots(nrows=num_results, ncols=1, sharex=True)
    fig.canvas.set_window_title('Testing Results')
    fig.suptitle("Test Predict Result Percentages")

    # Plot each row
    for row in range(num_results):

        row_results = [result_dict.get(label)[row] for label in result_dict.keys()]
        row_labels = list(result_dict.keys())

        # Ignore last entries, the 'actual_labels'
        row_results = row_results[:-1]
        row_labels = row_labels[:-1]

        print(row_results)
        print(row_labels)

        # Plot results
        bars = axes[row].bar(x=row_labels, height=row_results, color="gray")

        # Find the actual and predicted labels; change their colours
        predict_label = np.argmax(row_results)
        actual_label = result_dict.get("actual_class")[row]

        bars[predict_label].set_color('r')    # Predict overrides the gray
        bars[actual_label].set_color('b')     # If actual == predict, then this overrides the previous statement

        # Draw the mean as a horizontal line
        mean = sum(row_results) / len(row_results)
        axes[row].axhline(mean, color='k', linestyle='dashed', linewidth=1)



    # result_df = result_df.sample(frac=1).reset_index(drop=True)
    # results_to_plot = result_df.head(num)
    # # print(results_to_plot)
    #
    # # Get highest probability predicted and lock all figures's y axis to that
    # max_value = results_to_plot.iloc[[0, -1]].max()
    # max_value = max_value[0:-1].max()
    #

    #
    # # Plot each result
    # for index, row in results_to_plot.iterrows():
    #     # Find the predicted and actual labels
    #     # predict_label = np.argmax(row[:-1].values)
    #     # actual_label = xlabels.index(row.values[-1])

    #
    #     # plot_row(row, axes[index], results_to_plot.columns[0:-1], max_value, show_actual_label=True)

    plt.show()


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

    # Take a proportion of the data - there are just too many points to analyse
    # df_train = raw_df.head(20000)
    # df_test = raw_df.tail(10000)

    print("--- Training DF ---\n", df_train.head(50))
    print("--- Testing DF ---\n", df_test.head(50))

    train_features = pp_feature(df_train)
    test_features = pp_feature(df_test)
    
    train_targets = pp_targets(df_train)
    test_targets = pp_targets(df_test)

    classifier = train_model(train_features=train_features,
                             train_targets=train_targets,
                             test_features=test_features,
                             test_targets=test_targets,
                             learning_rate=0.0001,  # 0.003 works
                             steps=3000,
                             batch_size=100,  # 100 works
                             model_dir=model_dir_path)

    predictions = predict_model(classifier, test_features, test_targets)
    test_result_plotter(predictions, 4)


main()
