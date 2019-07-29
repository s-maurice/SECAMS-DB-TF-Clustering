from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import os
from sklearn import tree
from sklearn import neural_network
import matplotlib.pyplot as plt
# import graphviz


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


#  Read csv
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

# Possibly use one hot encoding here, however these are discrete but still linear, so encoding may not be too applicable

userid_label_encoder = preprocessing.LabelEncoder()
preprocessed_features['USERID'] = userid_label_encoder.fit_transform(preprocessed_features['USERID'])

present_label_encoder = preprocessing.LabelEncoder()
preprocessed_features['Present'] = present_label_encoder.fit_transform(raw_df['Present'])
# Be aware it assigns the first value it sees to 0, so present may not always be 1

for day in pd.to_datetime(raw_df['Day'].unique()):
    preprocessed_features.loc[(preprocessed_features['Day_of_month'] == day.day) & (preprocessed_features['Month_of_year'] == day.month), 'Users_absent'] = \
        len(preprocessed_features[(preprocessed_features['Day_of_month'] == day.day) & (preprocessed_features['Month_of_year'] == day.month) & (preprocessed_features['Present'] == present_label_encoder.transform([False])[0])]) / len(preprocessed_features['USERID'].unique())

# Process raw_df Labels
preprocessed_labels = pd.DataFrame()
reason_label_encoder = preprocessing.LabelEncoder()
preprocessed_labels['Reason'] = reason_label_encoder.fit_transform(raw_df['Reason'])
# Be aware it assigns the first value it sees to 0, possibly use one_hot instead

# print(preprocessed_labels)

# Split the training and testing data sets
train_labels, test_labels, train_features, test_features = train_test_split(preprocessed_labels,
                                                                            preprocessed_features,
                                                                            test_size=0.2)

# Begin Training
# Try to load the model if True
if os.path.isfile('saved_model.pkl'):
    classifier = joblib.load('saved_model.pkl')
else:
    # classifier = tree.DecisionTreeClassifier()  # Create Classifier, doesn't even need any of the params changed
    classifier = neural_network.MLPClassifier(verbose=True)  # DNN Classifier
    classifier.fit(train_features, train_labels)  # Fit Model
    # Save Model
    # Output a pickle file for the model
    joblib.dump(classifier, 'saved_model.pkl')


# Begin Testing
# Test model on test data set - can use .predict_proba instead, but the probability is always 1
test_predict_results = classifier.predict(test_features)
test_predict_results_proba = classifier.predict_proba(test_features)

# Convert into a Data Frame with matching features and actual_labels for easy analysis
test_result_df = test_features.copy()

test_result_df["Present"] = present_label_encoder.inverse_transform(test_result_df["Present"])  # Invert encoding
test_result_df["USERID"] = userid_label_encoder.inverse_transform(test_result_df["USERID"])

test_result_df["Actual Labels"] = reason_label_encoder.inverse_transform(test_labels)  # Invert encoding back into str
test_result_df["Predicted Labels"] = reason_label_encoder.inverse_transform(test_predict_results)

test_result_df.reset_index(drop=True, inplace=True)

test_result_proba_df = pd.DataFrame.from_records(test_predict_results_proba)
test_result_proba_df.columns = reason_label_encoder.inverse_transform(test_result_proba_df.columns)

print(test_result_proba_df)
print(test_result_df)

test_accuracy = classifier.score(test_features, test_labels)  # Gets mean accuracy of test data set
print("Accuracy:", test_accuracy)

print()
print("0-0-0-0-0-0-0-0-0-0")
print()

wrong_df = test_result_df[test_result_df['Actual Labels'] != test_result_df['Predicted Labels']]

wrong_proba_df = test_result_proba_df.loc[wrong_df.index]
wrong_proba_df['Actual Labels'] = wrong_df['Actual Labels']
wrong_proba_df['Predicted Labels'] = wrong_df['Predicted Labels']

print(wrong_proba_df)

num_cols = 5
num_rows = 5
fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, sharex=True, sharey=True)
for idx, row in wrong_proba_df.head(num_rows * num_cols).reset_index(drop=True).iterrows():
    bars = ax[idx // 5][idx % 5].bar(row.index[:-2], row[:-2], color='gray')

    predicted_label = reason_label_encoder.transform([row[-1]])[0]
    actual_label = reason_label_encoder.transform([row[-2]])[0]

    bars[predicted_label].set_color('r')
    bars[actual_label].set_color('b')

    mean = sum(row[:-2]) / len(row[:-2])
    ax[idx // 5][idx % 5].axhline(mean, color='k', linestyle='dashed', linewidth=1)


[plt.setp(item.get_xticklabels(), ha="right", rotation=90) for row in ax for item in row]
plt.show()

# Display the tree

# Doesn't use Graphvis
# plt.figure("Tree Graph", dpi=200)
# tree.plot_tree(classifier.fit(train_features, train_labels),
#                fontsize=2,
#                feature_names=preprocessed_features.columns,
#                class_names=reason_label_encoder.classes_)
# plt.show()

# Uses Graphvis - need to both download and pip
# tree_plot = tree.export_graphviz(classifier, out_file=None)
# tree_graph = graphviz.Source(tree_plot)
# tree_graph.render("Classifier_Tree")
