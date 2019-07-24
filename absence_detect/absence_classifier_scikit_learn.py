from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
# import graphviz


pd.set_option('display.max_columns', None)


#  Read csv
raw_df = pd.read_csv("reason_df.csv")
raw_df['Day'] = pd.to_datetime(raw_df['Day'])

# Process raw_df Features
preprocessed_features = pd.DataFrame()
preprocessed_features['USERID'] = [str(i) for i in raw_df['USERID']]
preprocessed_features['Day_of_week'] = [day.weekday() for day in raw_df['Day']]
preprocessed_features['Day_of_month'] = [day.day for day in raw_df['Day']]
preprocessed_features['Month_of_year'] = [day.month for day in raw_df['Day']]
# Possibly use one hot encoding here, however these are discrete but still linear, so encoding may not be too applicable

userid_label_encoder = preprocessing.LabelEncoder()
preprocessed_features['USERID'] = userid_label_encoder.fit_transform(preprocessed_features['USERID'])

present_label_encoder = preprocessing.LabelEncoder()
preprocessed_features['Present'] = present_label_encoder.fit_transform(raw_df['Present'])  
# Be aware it assigns the first value it sees to 0, so present may not always be 1

# print(preprocessed_features)

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
classifier = tree.DecisionTreeClassifier()  # Create Classifier, doesn't even need any of the params changed
classifier.fit(train_features, train_labels)  # Fit Model

# Begin Testing
# Test model on test data set - can use .predict_proba instead, but the probability is always 1
test_predict_results = classifier.predict(test_features)

# Convert into a Data Frame with matching features and actual_labels for easy analysis
test_result_df = test_features.copy()

test_result_df["Present"] = present_label_encoder.inverse_transform(test_result_df["Present"])  # Invert encoding back into str
test_result_df["USERID"] = userid_label_encoder.inverse_transform(test_result_df["USERID"])

test_result_df["Actual Labels"] = reason_label_encoder.inverse_transform(test_labels)  # Invert encoding back into str
test_result_df["Predicted Labels"] = reason_label_encoder.inverse_transform(test_predict_results)
print(test_result_df)

test_accuracy = classifier.score(test_features, test_labels)  # Gets mean accuracy of test data set
print("Accuracy:", test_accuracy)

# Display the tree

# Doesn't use Graphvis
plt.figure("Tree Graph", dpi=200)
tree.plot_tree(classifier.fit(train_features, train_labels),
               fontsize=2,
               feature_names=preprocessed_features.columns,
               class_names=reason_label_encoder.classes_)
plt.show()

# Uses Graphvis - need to both download and pip
# tree_plot = tree.export_graphviz(classifier, out_file=None)
# tree_graph = graphviz.Source(tree_plot)
# tree_graph.render("Classifier_Tree")
