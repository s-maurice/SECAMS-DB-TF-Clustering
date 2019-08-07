import itertools

import math
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import data
import os
from sklearn import preprocessing
import shutil
import datetime as dt


def predict_plot(proba_df, name=None, num_cols=5, num_rows=5):

    # Functions for plotting the features; allows easier reading
    def get_weekday(num):   # Function that returns a String of the day of week, given a number from .weekday()
        days_of_week = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        return days_of_week[num]

    def get_present(bool):  # Function that returns 'present' or 'absent' depending on the 'present' parameter
        if bool:
            return "Present"
        else:
            return "Absent"

    proba_df = proba_df.head(num_cols*num_rows)

    fig, ax = plt.subplots(figsize=(16, 8), nrows=num_rows, ncols=num_cols, sharex=True, sharey=True)
    for idx, row in proba_df.head(num_rows * num_cols).reset_index(drop=True).iterrows():
        # coordinates of the ax graph
        a = idx // 5
        b = idx % 5

        # Get labels and their heights; plot this as a bar graph
        labels = row.index[:14]
        heights = row[:14]
        bars = ax[a][b].bar(labels, heights, color='gray')

        # Find predicted and actual labels
        predicted_label = row['Predicted Labels']
        actual_label = row['Actual Labels']

        # Get indexes to set color
        predicted_index = proba_df.columns.tolist().index(row["Predicted Labels"])
        actual_index = proba_df.columns.tolist().index(row["Actual Labels"])
        bars[predicted_index].set_color('r')
        bars[actual_index].set_color('b')

        # Draw a mean line
        mean = sum(heights) / len(heights)
        ax[a][b].axhline(mean, color='k', linestyle='dashed', linewidth=1)

        # Find features and place them on the graph as well
        day = dt.date(year=2016, month=row['Month_of_year'], day=row['Day_of_month'])
        weekday = get_weekday(row['Day_of_week'])
        present = get_present(row['Present'])
        prev_absences = row['Prev_absences']
        users_absent = row['Users_absent']

        features = "%s (%s)\n%s / %s / %.3f\nP: %.3s / A: %.3s" % (day, weekday, present, prev_absences, users_absent, predicted_label, actual_label)

        # As a subtitle
        # font = dict(fontsize=8)
        # ax[a][b].set_title(features, fontdict=font, pad=-10)

        # As a textbox
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax[a][b].text(0.05, 0.95, features, fontsize=8, transform=ax[a][b].transAxes, verticalalignment='top', bbox=props)

    # Rotate x tick labels
    [plt.setp(item.get_xticklabels(), ha="center", rotation=90) for row in ax for item in row]
    fig.text(0.5, 0, 'Reason', ha='center', va='bottom')
    fig.text(0.06, 0.5, 'Percentage Confidence', ha='center', va='center', rotation='vertical')
    plt.subplots_adjust(left=0.09, bottom=0.15, top=0.94, wspace=0.06, hspace=0.09)
    if name is not None:
        fig.canvas.set_window_title(name)
        fig.suptitle(name)


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
    feature_df["Prev_absences"] = [int(i) for i in df["Prev_absences"]]
    feature_df["USERID"] = [str(i) for i in df["USERID"]]

    userid_le = preprocessing.LabelEncoder()
    feature_df["USERID"] = userid_le.fit_transform(feature_df["USERID"])

    for day in pd.to_datetime(df['Day'].unique()):
        feature_df.loc[(feature_df['Day_of_month'] == day.day) & (
                feature_df['Month_of_year'] == day.month), 'Users_absent'] = \
            len(feature_df[(feature_df['Day_of_month'] == day.day) & (
                    feature_df['Month_of_year'] == day.month) & (feature_df['Present'] ==
                                                                 one_hot_encode([False]))]) / len(feature_df['USERID'].unique())

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

    feature_column_list.append(tf.feature_column.numeric_column(key="Prev_absences"))

    feature_column_list.append(tf.feature_column.numeric_column(key="Users_absent"))

    feature_column_list.append(tf.feature_column.indicator_column(
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(key="USERID",
                                                                                     vocabulary_list=df[
                                                                                         "USERID"].unique())))

    return feature_column_list


def train_model(train_features, train_targets, test_features, test_targets, learning_rate, steps, batch_size, model_dir):
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

    periods = 1  # Training periods #TODO set back to 10
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
    predict_input_fn = lambda: create_input_function(features, targets, batch_size=1, num_epochs=1, shuffle=False)

    predict_results = classifier.predict(predict_input_fn)

    # # debug code
    # i = 0
    # for idx, prediction in enumerate(predict_results):
    #     if i == 10:
    #         break
    #     for key, value in prediction.items():
    #         print(key, ":", value)
    #     i += 1

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

    # print(result_dict)

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

    # df_train = raw_df.head(math.floor(len(raw_df) * 0.7))
    # df_test = raw_df.tail(math.floor(len(raw_df) * 0.3))

    # Take a proportion of the data - there are just too many points to analyse
    df_train = raw_df.head(20000)
    df_test = raw_df.tail(10000)

    print("--- Training DF ---\n", df_train.head(500))
    print("--- Testing DF ---\n", df_test.head(50))

    train_features = pp_feature(df_train)
    test_features = pp_feature(df_test)

    train_targets = pp_targets(df_train)
    test_targets = pp_targets(df_test)

    classifier = train_model(train_features=train_features,
                             train_targets=train_targets,
                             test_features=test_features,
                             test_targets=test_targets,
                             learning_rate=0.003,  # 0.003 works
                             steps=10,  # 3000
                             batch_size=1,  # 100 works
                             model_dir=model_dir_path)

    predictions = predict_model(classifier, test_features, test_targets)

    pd.set_option('display.max_columns', 15)
    pd.set_option('display.max_rows', 10)
    pd.set_option('display.width', None)

    # Place into DF for plotting
    predictions_labels_df = pd.DataFrame()
    for index, prediction in enumerate(predictions):
        if index == 25:
            break
        if index == 0:
            predictions_df = pd.DataFrame(columns=prediction.get("all_classes"))
            predictions_df.columns = [i.decode("utf-8") for i in predictions_df.columns]

        predictions_df.loc[index] = prediction.get("probabilities")
        # predictions_labels_df.loc[index, "Actual Labels"] = prediction.get("classes")[0].decode("utf-8")
        predictions_labels_df.loc[index, "Predicted Labels"] = predictions_df.columns.tolist()[np.argmax(prediction.get("probabilities"))]

    # print(predictions_df)
    # print(predictions_labels_df)
    # print(test_features.head(25))
    # print(test_targets.head(25))

    predictions_to_plot_df = pd.concat([predictions_df, test_features.head(25), predictions_labels_df, test_targets.head(25).reset_index(drop=True)], axis="columns")
    predictions_to_plot_df.rename(columns={"Reason": "Actual Labels"}, inplace=True)
    print(predictions_to_plot_df)
    predict_plot(predictions_to_plot_df, name="TensorFlow Predictions")
    plt.savefig('tensorflow_all.png', dpi=500)


main()
plt.show()
