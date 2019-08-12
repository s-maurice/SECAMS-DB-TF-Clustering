import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import pickle

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


# Gets object from pickles
def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)


# Plotter
def predict_plot(proba_df, name=None, plot_type=None, num_cols=5, num_rows=5):

    # Functions for writing features in a textbox; allows easier reading
    def get_weekday(num):   # Function that returns a String of the day of week, given a number from .weekday()
        days_of_week = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        return days_of_week[num]

    def get_present(bool):  # Function that returns 'present' or 'absent' depending on the 'present' parameter
        if bool:
            return "Present"
        else:
            return "Absent"
    
    # Choose the proba_df type:
    if plot_type == "correct":
        proba_df = proba_df[proba_df['Actual Labels'] == proba_df['Predicted Labels']]  # Correct entries
    elif plot_type == "incorrect":
        proba_df = proba_df[proba_df['Actual Labels'] != proba_df['Predicted Labels']]  # Wrong entries
    elif plot_type == "no_normal":
        proba_df = proba_df[proba_df['Actual Labels'] != "Normal"]  # Entries, apart from Normal
    elif plot_type == "no_normal_correct":
        proba_df = proba_df[(proba_df['Actual Labels'] != "Normal") & (proba_df['Actual Labels'] == proba_df['Predicted Labels'])]  # Correct entries, apart from Normal
    elif plot_type == "irregular":
        proba_df = proba_df[(proba_df['Actual Labels'] != "Normal") &
                            (proba_df['Actual Labels'] != "Holiday") &
                            (proba_df['Actual Labels'] != "Leave") &
                            (proba_df['Actual Labels'] != "Weekend_Work")]  # Entries that aren't Normal, Holiday, Weekend_Work or Leave

    proba_df = proba_df.head(num_cols * num_rows)

    fig, ax = plt.subplots(figsize=(16, 8), nrows=num_rows, ncols=num_cols, sharex=True, sharey=True)
    for idx, row in proba_df.head(num_rows * num_cols).reset_index(drop=True).iterrows():

        # coordinates of the ax graph
        a = idx // 5
        b = idx % 5

        # Get labels and their heights; plot this as a bar graph
        labels = reason_encoder.classes_
        heights = row[labels]
        bars = ax[a][b].bar(labels, heights, color='gray')

        # Lock scale
        ax[a][b].set_ylim(0, 1)

        # Find predicted and actual labels
        predicted_label = row['Predicted Labels']
        actual_label = row['Actual Labels']

        # Get indexes to set color
        predicted_index = reason_encoder.transform([predicted_label])[0]
        actual_index = reason_encoder.transform([actual_label])[0]
        bars[predicted_index].set_color('r')
        bars[actual_index].set_color('b')

        # Draw a mean line
        mean = sum(heights) / len(heights)
        ax[a][b].axhline(mean, color='k', linestyle='dashed', linewidth=1)

        # Lock axes
        ax[a][b].set_ylim(0, 1)

        # Find features and place them on the graph as well
        day = dt.date(year=2016, month=row['Month_of_year'], day=row['Day_of_month'])
        weekday = get_weekday(row['Day_of_week'])
        present = get_present(row['Present'])
        prev_absences = row['Prev_absences']
        users_absent = row['Users_absent']

        features = "%s (%s)\n%s / %s / %.3f\nP: %.3s / A: %.3s" % (day, weekday, present, prev_absences, users_absent, predicted_label, actual_label)

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


# Load features, labels and encoders; decode them
test_features = pd.read_csv("test_features.csv", index_col=0)
test_labels = pd.read_csv("test_labels.csv", index_col=0)
present_encoder = load_object("encoder_present.pkl")
userid_encoder = load_object("encoder_userid.pkl")
reason_encoder = load_object("encoder_reason.pkl")

test_features['USERID'] = userid_encoder.inverse_transform(test_features['USERID'])
test_features['Present'] = present_encoder.inverse_transform(test_features['Present'])

test_labels['Actual Labels'] = reason_encoder.inverse_transform(test_labels['Reason'])


# --- SKLEARN ---
sklearn_prob_df = pd.read_csv("predictions_sklearn.csv", index_col=0)

# Decode column names
sklearn_prob_df.columns = reason_encoder.inverse_transform([int(i) for i in sklearn_prob_df.columns])

# Get predicted labels
sklearn_prob_df['Predicted Labels'] = sklearn_prob_df.idxmax(axis=1)

# Add on features + actual labels
sklearn_prob_df = pd.concat([sklearn_prob_df, test_features, test_labels], axis=1)


# --- TENSORFLOW ---
tf_prob_df = pd.read_csv("tensorflow predictions df", index_col=0)
tf_prob_df.columns = [int(i) for i in tf_prob_df.columns]
tf_prob_df = tf_prob_df.reindex(sorted(tf_prob_df.columns), axis=1)

tf_prob_df.columns = reason_encoder.inverse_transform(tf_prob_df.columns)
tf_prob_df['Predicted Labels'] = tf_prob_df.idxmax(axis=1)

tf_prob_df = pd.concat([tf_prob_df, test_features, test_labels], axis=1)

# Plotting
plot_types = ["correct", "incorrect", "no_normal", "no_normal_correct", "irregular"]
for plot_type in plot_types:
    predict_plot(sklearn_prob_df, name="Scikit Learn (" + plot_type + ")", plot_type=plot_type)
    predict_plot(tf_prob_df, name="TensorFlow (" + plot_type + ")", plot_type=plot_type)

plt.show()
