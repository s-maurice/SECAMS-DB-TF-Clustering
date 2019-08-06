import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)

# Only 14 users, 4000ish lines in this csv
raw_df = pd.read_csv("CSV Files/Curated Data/ALL_USERID_beginning_with_20_and_between_100_and_500_entries.csv")
raw_df['TIMESTAMPS'] = pd.to_datetime(raw_df['TIMESTAMPS'])


# Process Features
preprocessed_features = pd.DataFrame()
preprocessed_features["UserID"] = raw_df["USERID"]
preprocessed_features["TerminalSN"] = raw_df["TERMINALSN"]
preprocessed_features["Time_Of_Day"] = [timestamp.hour + timestamp.minute / 60 for timestamp in raw_df["TIMESTAMPS"]]
preprocessed_features["Day_Of_Month"] = [timestamp.day for timestamp in raw_df["TIMESTAMPS"]]
preprocessed_features["Day_Of_Week"] = [timestamp.weekday() for timestamp in raw_df["TIMESTAMPS"]]
# preprocessed_features["EventID"] = raw_df["EVENTID"]


# Label Encode Features
UserID_le = preprocessing.LabelEncoder()
preprocessed_features["UserID"] = UserID_le.fit_transform(preprocessed_features["UserID"])

TerminalSN_le = preprocessing.LabelEncoder()
preprocessed_features["TerminalSN"] = TerminalSN_le.fit_transform(preprocessed_features["TerminalSN"])

# EventID_le = preprocessing.LabelEncoder()
# preprocessed_features["EventID"] = EventID_le.fit_transform(preprocessed_features["EventID"])


# Split data set, not needed as it is unsupervised
# train_features, test_features = train_test_split(preprocessed_features, test_size=0.2)

# Begin Training
neigh = LocalOutlierFactor(novelty=False)  # Default Args
train_outliers = neigh.fit_predict(preprocessed_features)  # On training data

outlier_result_df = pd.DataFrame()
outlier_result_df["UserID"] = UserID_le.inverse_transform(preprocessed_features["UserID"])
outlier_result_df["TerminalSN"] = TerminalSN_le.inverse_transform(preprocessed_features["TerminalSN"])
outlier_result_df["Timestamps"] = raw_df["TIMESTAMPS"]
outlier_result_df["Outlier"] = train_outliers
print(outlier_result_df)

