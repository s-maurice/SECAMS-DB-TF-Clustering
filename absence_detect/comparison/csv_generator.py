import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle

# This file generates CSV files for training and testing.
# Will be used to compare the specific results of ML models from TensorFlow vs SciKit learn.

pd.set_option('display.max_rows', 10)

raw_df = pd.read_csv("../reason_df.csv")

raw_df['Day'] = pd.to_datetime(raw_df['Day'])


# --- features ---
pp_features = pd.DataFrame()
pp_features['USERID'] = [str(i) for i in raw_df['USERID']]
pp_features['Day_of_week'] = [day.weekday() for day in raw_df['Day']]
pp_features['Day_of_month'] = [day.day for day in raw_df['Day']]
pp_features['Month_of_year'] = [day.month for day in raw_df['Day']]
pp_features['Prev_absences'] = raw_df['Prev_absences']

userid_label_encoder = preprocessing.LabelEncoder()
pp_features['USERID'] = userid_label_encoder.fit_transform(pp_features['USERID'])
present_label_encoder = preprocessing.LabelEncoder()
pp_features['Present'] = present_label_encoder.fit_transform(raw_df['Present'])
# Be aware it assigns the first value it sees to 0, so present may not always be 1

for day in pd.to_datetime(raw_df['Day'].unique()):
    pp_features.loc[(pp_features['Day_of_month'] == day.day) & (pp_features['Month_of_year'] == day.month), 'Users_absent'] = \
        len(pp_features[(pp_features['Day_of_month'] == day.day) &
                        (pp_features['Month_of_year'] == day.month) &
                        (pp_features['Present'] == present_label_encoder.transform([False])[0])]) \
        / len(pp_features['USERID'].unique())


# --- labels ---
pp_labels = pd.DataFrame()
reason_label_encoder = preprocessing.LabelEncoder()
pp_labels['Reason'] = reason_label_encoder.fit_transform(raw_df['Reason'])

# split
train_labels, test_labels, train_features, test_features = train_test_split(pp_labels, pp_features, test_size=0.3)

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

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


save_object(present_label_encoder, "encoder_present.pkl")
save_object(userid_label_encoder, "encoder_userid.pkl")
save_object(reason_label_encoder, "encoder_reason.pkl")
