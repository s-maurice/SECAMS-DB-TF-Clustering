from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import os
from sklearn import tree
from sklearn import neural_network
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RBF
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
# import graphviz


pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', None)

# Hyperparameters?
# WARNING: The test_features.csv file generated while one_hot_encode is set to True requires a lot of memory.
one_hot_encode = False
max_iter = 200    # (200 by default)
classifier_type = "Cal_Class_Test"  # Classifiers: Tree / DNN / Gaussian / Cal_Class_Test # TODO set to cal_class_test
use_calibrator = True               # Whether to calibrate (always, if using 'Cal_Class_Test') # TODO set to True
calibration_type = "sigmoid"        # Sigmoid / Isotonic
early_stopping = True               # Stop based on validation acc, rather than loss (disabled, if using 'Cal_Class_Test')
show_examples = True                # Show example predictions


# Preprocess the data and encode the features + labels
# Read csv
raw_df = pd.read_csv("reason_df.csv")
raw_df['Day'] = pd.to_datetime(raw_df['Day'])

# Process raw_df Features
# features = ['USERID', 'Day_of_week', '']

preprocessed_features = pd.DataFrame()
preprocessed_features['USERID'] = [str(i) for i in raw_df['USERID']]
preprocessed_features['Day_of_week'] = [day.weekday() for day in raw_df['Day']]
preprocessed_features['Day_of_month'] = [day.day for day in raw_df['Day']]
preprocessed_features['Month_of_year'] = [day.month for day in raw_df['Day']]
preprocessed_features['Prev_absences'] = raw_df['Prev_absences']
# Add feature for whether they drive a car or take the train to work

present_label_encoder = preprocessing.LabelEncoder()
preprocessed_features['Present'] = present_label_encoder.fit_transform(raw_df['Present'])
# Be aware it assigns the first value it sees to 0, so present may not always be 1

for day in pd.to_datetime(raw_df['Day'].unique()):
    preprocessed_features.loc[(preprocessed_features['Day_of_month'] == day.day) & (
                preprocessed_features['Month_of_year'] == day.month), 'Users_absent'] = \
        len(preprocessed_features[(preprocessed_features['Day_of_month'] == day.day) & (
                    preprocessed_features['Month_of_year'] == day.month) & (preprocessed_features['Present'] ==
                                                                            present_label_encoder.transform(
                                                                                [False])[0])]) / len(
            preprocessed_features['USERID'].unique())

# One-hot encode USERID through get_dummies; concat this to DF and drop original USERID column
if one_hot_encode:
    userid_encoded_df = pd.get_dummies(preprocessed_features['USERID'].to_list())
    preprocessed_features = pd.concat([preprocessed_features, userid_encoded_df], axis=1)
    preprocessed_features.drop(["USERID"], inplace=True, axis=1)
else:
    userid_label_encoder = preprocessing.LabelEncoder()
    preprocessed_features['USERID'] = userid_label_encoder.fit_transform(preprocessed_features['USERID'])

print("Preprocessed feature sample:\n", preprocessed_features.head(10))

# Process raw_df Labels
preprocessed_labels = pd.DataFrame()
reason_label_encoder = preprocessing.LabelEncoder()
preprocessed_labels['Reason'] = reason_label_encoder.fit_transform(raw_df['Reason'])
# Be aware it assigns the first value it sees to 0, possibly use one_hot instead

# Begin Training: Try to load the model and test data sets if the model already exists
if os.path.isfile('saved_model.pkl'):
    classifier = joblib.load('saved_model.pkl')
    test_labels = pd.read_csv('test_labels.csv', index_col=0)
    test_features = pd.read_csv('test_features.csv', index_col=0)

else:
    # Split the training and testing data sets; save this test data set for later use too
    train_labels, test_labels, train_features, test_features = train_test_split(preprocessed_labels, preprocessed_features, test_size=0.2)

    test_features.to_csv("test_features.csv")
    test_labels.to_csv("test_labels.csv")

    # Create one of the following classifiers:
    if classifier_type == "Tree":
        classifier = tree.DecisionTreeClassifier()  # Create Classifier, doesn't even need any of the params changed
    elif classifier_type == "DNN":
        classifier = neural_network.MLPClassifier(verbose=True, max_iter=max_iter, early_stopping=early_stopping, hidden_layer_sizes=(100, 50))
    elif classifier_type == "Gaussian":
        classifier = gaussian_process.GaussianProcessClassifier(kernel=1.0*RBF(1.0))
    elif classifier_type == "Cal_Class_Test":
        classifier = neural_network.MLPClassifier(verbose=True, max_iter=max_iter, hidden_layer_sizes=(100, 50))
        classifier = CalibratedClassifierCV(classifier, cv=5, method="isotonic")
        use_calibrator = False    # already calibrated
    else:   # Default to DNN
        print("Classifier Unselected, Defaulting to DNN")
        print("----------------------------------------")
        classifier = neural_network.MLPClassifier(verbose=True, max_iter=max_iter, hidden_layer_sizes=(100, 50))

    classifier.fit(train_features, train_labels)  # Fit Model

    if use_calibrator:      # Calibrate Classifier to adjust label probabilities
        print('calibrating...')
        classifier = CalibratedClassifierCV(classifier, cv="prefit", method=calibration_type)  # Defaults to sigmoid
        classifier.fit(train_features, train_labels)
        print('calibrated')

    # Save Model as a pickle file
    joblib.dump(classifier, 'saved_model.pkl')


# Begin Testing
# Test model on test data set - can use .predict_proba instead, but the probability is always 1
print("Test feature sample:\n", test_features.head(10))
test_predict_results = classifier.predict(test_features)
test_predict_results_proba = classifier.predict_proba(test_features)

# Convert into a Data Frame with matching features and actual_labels for easy analysis
test_result_df = test_features.copy()

# Invert encoding
test_result_df["Present"] = present_label_encoder.inverse_transform(test_result_df["Present"])
if not one_hot_encode:
    test_result_df["USERID"] = userid_label_encoder.inverse_transform(test_result_df["USERID"])

test_result_df["Actual Labels"] = reason_label_encoder.inverse_transform(test_labels)  # Invert encoding back into str
test_result_df["Predicted Labels"] = reason_label_encoder.inverse_transform(test_predict_results)

test_result_df.reset_index(drop=True, inplace=True)

test_result_proba_df = pd.DataFrame.from_records(test_predict_results_proba)
test_result_proba_df.columns = reason_label_encoder.inverse_transform(test_result_proba_df.columns)

print("Test result sample:\n", test_result_df.head(10))

test_accuracy = classifier.score(test_features, test_labels)  # Gets mean accuracy of test data set
print("Accuracy:", test_accuracy)

test_results = pd.concat([test_result_proba_df, test_result_df], axis=1)

# Create a number of DFs from the probabilities:
correct_proba_df = test_results[test_results['Actual Labels'] == test_results['Predicted Labels']]  # Correct entries
no_normal_correct_proba_df = correct_proba_df[correct_proba_df['Actual Labels'] != "Normal"]        # Correct entries, apart from Normal
wrong_proba_df = test_results[test_results['Actual Labels'] != test_results['Predicted Labels']]    # Wrong entries
no_normal_proba_df = test_results[test_results['Actual Labels'] != "Normal"]                        # Entries, apart from Normal
irregular_proba_df = test_results[(test_results['Actual Labels'] != "Normal") &
                                  (test_results['Actual Labels'] != "Holiday") &
                                  (test_results['Actual Labels'] != "Leave") &
                                  (test_results['Actual Labels'] != "Weekend_Work")]  # Entries that aren't Normal, Holiday, Weekend_Work or Leave


def average_actual_deviation(proba_df):  # Calculates how far off the model is on average
    pred = proba_df["Predicted Labels"].to_list()
    actual = proba_df["Actual Labels"].to_list()

    deviation = []
    for (index, row), zero_index in zip(proba_df.iterrows(), range(len(proba_df))):
        deviation.append(row[pred[zero_index]] - row[actual[zero_index]])
    return np.mean(np.abs(deviation))


def prediction_actual_hist(proba_df_list, name_list):  # Shows hist of predicted values v actual values occurrences
    fig, ax = plt.subplots(len(proba_df_list), 2, sharey=True, sharex=True, figsize=(16, 8))
    fig.canvas.set_window_title('Predicted vs. Actual Labels')

    for index, proba_df in zip(range(len(proba_df_list)), proba_df_list):
        pred = proba_df["Predicted Labels"].to_list()
        actual = proba_df["Actual Labels"].to_list()

        ax[index, 0].set_ylabel(name_list[index], rotation=90, size="large")
        ax[index, 0].hist(pred)
        ax[index, 1].hist(actual)

    plt.setp(ax[-1, 0].get_xticklabels(), ha="center", rotation=90)
    plt.setp(ax[-1, 1].get_xticklabels(), ha="center", rotation=90)

    ax[0, 0].set_title("Predicted Labels")
    ax[0, 1].set_title("Actual Labels")


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

        # Lock scale
        ax[a][b].set_ylim(0, 1)

        # Find predicted and actual labels
        predicted_label = row['Predicted Labels']
        actual_label = row['Actual Labels']

        # Get indexes to set color
        predicted_index = reason_label_encoder.transform([predicted_label])[0]
        actual_index = reason_label_encoder.transform([actual_label])[0]
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


print("---------")
print("Deviation")
print("Overall Deviation ", average_actual_deviation(test_results))
print("Wrong Average Deviation ", average_actual_deviation(wrong_proba_df))
print("No Normal Average Deviation ", average_actual_deviation(no_normal_proba_df))
print("Irregular Average Deviation ", average_actual_deviation(irregular_proba_df))

prediction_actual_hist([wrong_proba_df, no_normal_proba_df], ["wrong_proba_df", "no_normal_proba_df"])
# prediction_actual_hist([wrong_proba_df, no_normal_proba_df, irregular_proba_df], ["wrong_proba_df", "no_normal_proba_df", "irregular_proba_df"])

if show_examples:
    predict_plot(no_normal_proba_df, name="Results (excluding Normal)")
    predict_plot(irregular_proba_df, name="Results (excluding Normal, Leave, Weekend, Holiday)")
    plt.savefig('iregular.png', dpi=500)
    predict_plot(wrong_proba_df, name="Wrong Predictions")
    plt.savefig('wrong.png', dpi=500)
    predict_plot(correct_proba_df, name="Correct Predictions")
    plt.savefig('correct.png', dpi=500)
    predict_plot(no_normal_correct_proba_df, name="Correct Predictions (excluding Normal)")
    plt.savefig('no_normal_correct.png', dpi=500)

# Save wrong_proba_df for later analysis as well
wrong_proba_df.to_csv("wrong_proba_df.csv")

# If tree classifier, display the tree
if classifier_type == "Tree":
    # Doesn't use Graphvis
    plt.figure("Tree Graph", dpi=200)
    tree.plot_tree(classifier.fit(train_features, train_labels),
                   fontsize=2,
                   feature_names=preprocessed_features.columns,
                   class_names=reason_label_encoder.classes_)

    # # Uses Graphvis - need to both download and pip
    # tree_plot = tree.export_graphviz(classifier, out_file=None)
    # tree_graph = graphviz.Source(tree_plot)
    # tree_graph.render("Classifier_Tree")

# Hyperparam Printout

print("----------------")
print("Hyperparameters")
print("One Hot Encode: %s \nMax Iterations: %s \nClassifier Type: %s \nUse Calibrator: %s (%s) \nEarly Stopping: %s \nShow Examples: %s " % (one_hot_encode, max_iter, classifier_type, use_calibrator, calibration_type, early_stopping, show_examples))

plt.show()

