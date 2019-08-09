import pandas as pd
from sklearn.model_selection import train_test_split

# This file generates CSV files for training and testing.
# Will be used to compare the specific results of ML models from TensorFlow vs SciKit learn.

pd.set_option('display.max_rows', 10)

raw_df = pd.read_csv("../reason_df.csv")

raw_features = raw_df.drop(columns=['Reason'])
raw_labels = raw_df[['Reason']].copy()

print(raw_features)
print(raw_labels)

train_labels, test_labels, train_features, test_features = train_test_split(raw_labels, raw_features, test_size=0.3)

train_labels.reset_index(drop=True, inplace=True)
test_labels.reset_index(drop=True, inplace=True)
train_features.reset_index(drop=True, inplace=True)
test_features.reset_index(drop=True, inplace=True)

print(train_labels)
print(test_labels)
print(train_features)
print(test_features)

train_labels.to_csv("train_labels.csv")
test_labels.to_csv("test_labels.csv")
train_features.to_csv("train_features.csv")
test_features.to_csv("test_features.csv")
