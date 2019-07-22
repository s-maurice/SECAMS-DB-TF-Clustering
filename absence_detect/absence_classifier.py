import math
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import data


def pp_feature(df):
    feature_df = pd.DataFrame()
    # Take the following features of the DF: USERID, Day, Present
    feature_df['USERID'] = df['USERID']
    feature_df['Present'] = df['Present']

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


def train_model()


def main():
    raw_df = pd.read_csv("reason_df.csv")
    raw_df['Day'] = pd.to_datetime(raw_df['Day'])
    raw_df = raw_df.reindex(np.random.permutation(raw_df.index))
    raw_df.reset_index(inplace=True)

    df_train = raw_df.head(math.floor(len(raw_df) * 0.7))
    df_test = raw_df.tail(math.floor(len(raw_df) * 0.3))

    train_features = pp_feature(df_train)
    test_features = pp_feature(df_test)
    
    train_targets = pp_targets(df_train)
    test_targets = pp_targets(df_test)

    train_if = create_input_function(train_features, train_targets, batch_size)

    print(train_features)
    print(train_targets)


main()