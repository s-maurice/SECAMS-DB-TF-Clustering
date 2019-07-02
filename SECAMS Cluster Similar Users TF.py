import numpy as np
import pandas as pd
import tensorflow as tf
import pyodbc
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as svm

import get_input_data


def day_time_normed(events):
    decimal_hour = (events.dt.hour + events.dt.minute / 60)

    tx = decimal_hour.apply(lambda x: np.math.cos(x * np.pi / 12))
    ty = decimal_hour.apply(lambda x: np.math.sin(x * np.pi / 12))
    timexy = pd.DataFrame([tx, ty]).transpose()
    timexy.columns = ["daytx", "dayty"]

    return timexy


def dataPreprocessing(df):
    # In the case of SECAMS DB, all terminals are on thr same TERMINALGROUP, so drop as it may confuse the DNN
    df = df.drop(columns=["TERMINALGROUP", "TERMINALNAME"])  # TerminalName redundant as same as teminal id FUTURE USE MAY CONSIDER GROUPING BOYS/GIRLS VERSIONS OF SCHOOLS

    # Modify TIMESTAMPS to "day of week" "month of year" <- catagorical, and "normedtime x/y" as dense
    # normedtime x/y
    timexy = day_time_normed(df["TIMESTAMPS"])
    df = df.join(timexy)
    df['DAYTIME'] = df[['daytx', 'dayty']].values.tolist()  # puts daytx and dayty into a single df column, with array dimension (2,1)
    # day of week
    dow = df["TIMESTAMPS"].dt.strftime("%a")
    df["DAYOFWEEK"] = dow
    # month of year
    moy = df["TIMESTAMPS"].dt.strftime("%b")
    df["MONTHOFYEAR"] = moy

    # Shuffle DF Rows
    df = df.sample(frac=1).reset_index(drop=True)  # could use shuffle(df) from sklearn.utils - better because can use stratify which garuanties good ratio of points in test/train data
    df_size = len(df.index)
    # Select ratio of train to test data - NEEDS REMAKE to handle decimal, give train_test_ratio differently - TEMP
    train_test_ratio = (4, 5)

    train_len = int(df_size * train_test_ratio[0] / train_test_ratio[1])
    test_len = int(df_size - (df_size * train_test_ratio[0] / train_test_ratio[1]))

    df_train = df.head(train_len).drop(["TIMESTAMPS"], axis=1)
    df_test = df.tail(test_len).drop(["TIMESTAMPS"], axis=1)

    return df_train, df_test


def define_feature_columns(dataset):
    sparse_df = dataset.drop(["daytx", "dayty"], axis=1).reset_index(drop=True) # Not nessacary, prep for iterable in future?

    # Create Feature Columns with each possible value for sparse data rows
    USERID_fc = tf.feature_column.categorical_column_with_vocabulary_list(key='USERID', vocabulary_list=sparse_df["USERID"].unique(), default_value=0)
    EVENTID_fc = tf.feature_column.categorical_column_with_vocabulary_list(key='EVENTID', vocabulary_list=sparse_df["EVENTID"].unique(), default_value=0)
    TERMINALSN_fc = tf.feature_column.categorical_column_with_vocabulary_list(key='TERMINALSN', vocabulary_list=sparse_df["TERMINALSN"].unique(), default_value=0)
    DOW_fc = tf.feature_column.categorical_column_with_vocabulary_list(key='DAYOFWEEK', vocabulary_list=sparse_df["DAYOFWEEK"].unique(), default_value=0)
    MOY_fc = tf.feature_column.categorical_column_with_vocabulary_list(key='MONTHOFYEAR', vocabulary_list=sparse_df["MONTHOFYEAR"].unique(), default_value=0)
    # numeric for timescale
    daytime_fc = tf.feature_column.numeric_column(key="DAYTIME", shape=[2, 1])  # put both daytx and dayty in as array, double check the shape



    # Wrap within an embedding column or indicator column
    # Sparse feature columns
    USERID_ic = tf.feature_column.indicator_column(categorical_column=USERID_fc) # dimension should be about number_of_categories**0.25 according to google, for embedding columns
    EVENTID_ic = tf.feature_column.indicator_column(categorical_column=EVENTID_fc) #indicator or embedding column???
    TERMINALSN_ic = tf.feature_column.indicator_column(categorical_column=TERMINALSN_fc)
    DOW_ic = tf.feature_column.indicator_column(categorical_column=DOW_fc)
    MOY_ic= tf.feature_column.indicator_column(categorical_column=MOY_fc)
    # #dense feature columns - need to embed?

    feature_columns_list = [USERID_ic, EVENTID_ic, TERMINALSN_ic, DOW_ic, MOY_ic, daytime_fc]


    # CategoricalColumn

    return feature_columns_list


def DNNBuilder(fc_list):
    # Build DNN Classifier - #USE DNNRegressor or DNNClassifier
    classifier = tf.estimator.DNNRegressor(feature_columns=fc_list, hidden_units=[256, 32])  # Not sure how many hidden units, layers/size, need more research/expermentation
    return classifier


def create_train_input_fn(df):
    # Places data into estimator
    input_fn = tf.estimator.inputs.pandas_input_fn(df, y=df["DAYTIME"], shuffle=True)  # other params needed? # Deprecated, use tf.compat.v1.estimator.inputs.pandas_input_fn instead
    return input_fn


def main():
    event_df = get_input_data.get_events()
    df_train, df_test = dataPreprocessing(event_df)

    fc_list = define_feature_columns(df_train)
    classifier = DNNBuilder(fc_list)
    input_fn = create_train_input_fn(df_train)

    classifier.train(input_fn=input_fn) # error likely means normalisation wasn't correctly done


main()
