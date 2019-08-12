import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Gets object from pickles
def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)


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


def filter_proba_dfs(proba_df):
    # Wrong entries
    incorrect_proba_df = proba_df[proba_df['Actual Labels'] != proba_df['Predicted Labels']]
    # Entries, apart from Normal
    no_normal_proba_df = proba_df[proba_df['Actual Labels'] != "Normal"]
    # Entries that aren't Normal, Holiday, Weekend_Work or Leave
    irregular_proba_df = proba_df[(proba_df['Actual Labels'] != "Normal") &
                                  (proba_df['Actual Labels'] != "Holiday") &
                                  (proba_df['Actual Labels'] != "Leave") &
                                  (proba_df['Actual Labels'] != "Weekend_Work")]
    return proba_df, incorrect_proba_df, no_normal_proba_df, irregular_proba_df


def get_deviation(proba_df_list):  # Calculates how far off the model is on average
    deviation_list = []
    for proba_df in proba_df_list:
        pred = proba_df["Predicted Labels"].to_list()
        actual = proba_df["Actual Labels"].to_list()

        deviation = []
        for (index, row), zero_index in zip(proba_df.iterrows(), range(len(proba_df))):
            deviation.append(row[pred[zero_index]] - row[actual[zero_index]])
        deviation_list.append(np.mean(np.abs(deviation)))
    return deviation_list


def get_accuracy(proba_df_list):
    acc_list = []
    for proba_df in proba_df_list:
        acc_list.append(len(proba_df[proba_df["Actual Labels"] == proba_df["Predicted Labels"]]) / len(proba_df))
    return acc_list


# List of filters
filter_list = ["All Entries", "Incorrect Entries", "Entries excluding Normal", "Irregular Entries"]

# Plot accuracy
tf_acc_list = get_accuracy(filter_proba_dfs(tf_prob_df))
sklearn_acc_list = get_accuracy(filter_proba_dfs(sklearn_prob_df))

fig, ax = plt.subplots(figsize=(16, 8))
ind = np.arange(3)    # the x locations for the groups
width = 0.35         # the width of the bars
p1 = ax.bar(ind, tf_acc_list[0:1] + tf_acc_list[2:], width, color="#ff6f00")
p2 = ax.bar(ind + width, sklearn_acc_list[0:1] + sklearn_acc_list[2:], width, color="#3499CD")

ax.set_xticks(ind + width / 2)
ax.set_xticklabels(filter_list[0:1] + filter_list[2:])
ax.set_ylim(0, 1)
ax.set_yticks(np.arange(0, 1, 0.1))
ax.set_ylabel("Percentage Accuracy")
ax.set_title("TensorFlow v. SciKit Learn Accuracy")
plt.legend((p1[0], p2[0]), ("TensorFlow", "SciKit Learn"))
ax.grid(axis="y")
fig.canvas.set_window_title("Accuracy Comparison")
plt.savefig('Accuracy Comparison.png', dpi=500)

# Plot deviation
tf_deviation_list = get_deviation(filter_proba_dfs(tf_prob_df))
sklearn_deviation_list = get_deviation(filter_proba_dfs(sklearn_prob_df))

fig2, ax = plt.subplots(figsize=(16, 8))
ind = np.arange(4)    # the x locations for the groups
width = 0.35         # the width of the bars
p1 = ax.bar(ind, tf_deviation_list, width, color="#ff6f00")
p2 = ax.bar(ind + width, sklearn_deviation_list, width, color="#3499CD")

ax.set_xticks(ind + width / 2)
ax.set_xticklabels(filter_list)
ax.set_ylim(0, 1)
ax.set_yticks(np.arange(0, 1, 0.1))
ax.set_ylabel("Mean Probability Deviation")
ax.set_title("TensorFlow v. SciKit Learn Probability Deviation")
plt.legend((p1[0], p2[0]), ("TensorFlow", "SciKit Learn"))
ax.grid(axis="y")
fig2.canvas.set_window_title("Deviation Comparison")
plt.savefig('Deviation Comparison.png', dpi=500)


plt.show()
