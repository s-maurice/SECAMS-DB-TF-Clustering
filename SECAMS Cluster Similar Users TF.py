import numpy as np
import pandas as pd
import tensorflow as tf
# import pyodbc
# import matplotlib.pyplot as plt
# import seaborn as sns
# import sklearn as svm

import get_input_data


def get_decimal_hour(events):
    decimal_hour = (events.dt.hour + events.dt.minute / 60)
    return decimal_hour


def day_time_normed(events):
    decimal_hour = get_decimal_hour(events)

    tx = decimal_hour.apply(lambda x: np.math.cos(x * np.pi / 12))
    ty = decimal_hour.apply(lambda x: np.math.sin(x * np.pi / 12))
    timexy = pd.DataFrame([tx, ty]).transpose()
    timexy.columns = ["daytx", "dayty"]

    return timexy
    # Note: Why turn time of day into 2 variables, rather than simply letting the ML model handle it?


def data_preprocessing(df):
    # In the case of SECAMS DB, all terminals are on thr same TERMINALGROUP, so drop as it may confuse the DNN
    df = df.drop(columns=["TERMINALGROUP", "TERMINALNAME"])  # TerminalName redundant as same as teminal id FUTURE USE MAY CONSIDER GROUPING BOYS/GIRLS VERSIONS OF SCHOOLS

    # Drop NANs
    df = df.dropna(how="any", axis=0)

    # Modify TIMESTAMPS to "day of week" "month of year" <- catagorical, and "normedtime x/y" as dense

    # This section uses daytimex daytimey in array, shape (2,1)
    # normedtime x/y
    #timexy = day_time_normed(df["TIMESTAMPS"])
    #df = df.join(timexy)
    #df['DAYTIME'] = df[['daytx', 'dayty']].values.tolist()  # puts daytx and dayty into a single df column, with array dimension (2,1)

    # This section uses decimal hour time / 24
    df["DECHOUR"] = get_decimal_hour(df["TIMESTAMPS"]).apply(lambda x: x / 24)

    print(df.columns)

    # day of week
    dow = df["TIMESTAMPS"].dt.strftime("%a")
    df["DAYOFWEEK"] = dow
    # month of year
    moy = df["TIMESTAMPS"].dt.strftime("%b")
    df["MONTHOFYEAR"] = moy

    # Shuffle DF Rows
    df = df.sample(frac=1).reset_index(drop=True)  # could use shuffle(df) from sklearn.utils - better because can use stratify which garuanties good ratio of points in test/train data

    # Select ratio of train to test data - NEEDS REMAKE to handle decimal, give train_test_ratio differently - TEMP
    df_size = len(df.index)
    train_test_ratio = (4, 5)

    train_len = int(df_size * train_test_ratio[0] / train_test_ratio[1])
    test_len = int(df_size - (df_size * train_test_ratio[0] / train_test_ratio[1]))

    df_train = df.head(train_len).drop(["TIMESTAMPS"], axis=1)
    df_test = df.tail(test_len).drop(["TIMESTAMPS"], axis=1)

    return df_train, df_test


def define_feature_columns(dataset):
    #sparse_df = dataset.drop(["daytx", "dayty"], axis=1).reset_index(drop=True) # Not necessary, prep for iterable in future?
    sparse_df = dataset

    # Create Feature Columns with each possible value for sparse data rows
    USERID_fc = tf.feature_column.categorical_column_with_vocabulary_list(key='USERID', vocabulary_list=sparse_df["USERID"].unique(), default_value=0)
    EVENTID_fc = tf.feature_column.categorical_column_with_vocabulary_list(key='EVENTID', vocabulary_list=sparse_df["EVENTID"].unique(), default_value=0)
    TERMINALSN_fc = tf.feature_column.categorical_column_with_vocabulary_list(key='TERMINALSN', vocabulary_list=sparse_df["TERMINALSN"].unique(), default_value=0)
    DOW_fc = tf.feature_column.categorical_column_with_vocabulary_list(key='DAYOFWEEK', vocabulary_list=sparse_df["DAYOFWEEK"].unique(), default_value=0)
    MOY_fc = tf.feature_column.categorical_column_with_vocabulary_list(key='MONTHOFYEAR', vocabulary_list=sparse_df["MONTHOFYEAR"].unique(), default_value=0)
    # numeric for timescale
    #daytime_fc = tf.feature_column.numeric_column(key="DECHOUR", shape=[1, 1])  # put both daytx and dayty in as array, double check the shape



    # Wrap within an embedding column or indicator column
    # Sparse feature columns
    USERID_ic = tf.feature_column.indicator_column(categorical_column=USERID_fc) # dimension should be about number_of_categories**0.25 according to google, for embedding columns
    EVENTID_ic = tf.feature_column.indicator_column(categorical_column=EVENTID_fc) #indicator or embedding column???
    TERMINALSN_ic = tf.feature_column.indicator_column(categorical_column=TERMINALSN_fc)
    DOW_ic = tf.feature_column.indicator_column(categorical_column=DOW_fc)
    MOY_ic= tf.feature_column.indicator_column(categorical_column=MOY_fc)
    # #dense feature columns - need to embed?

    feature_columns_list = [USERID_ic, EVENTID_ic, TERMINALSN_ic, DOW_ic, MOY_ic]

    return feature_columns_list


def DNNBuilder(fc_list):
    # Build DNN Classifier - #USE DNNRegressor or DNNClassifier
    classifier = tf.estimator.DNNRegressor(feature_columns=fc_list, hidden_units=[256, 32])  # Not sure how many hidden units, layers/size, need more research/expermentation
    return classifier


def create_input_fn(df):
    # Places data into estimator
    input_fn = tf.estimator.inputs.pandas_input_fn(df, y=df["DECHOUR"], shuffle=True)  # other params needed?, shuffle = true? # Deprecated, use tf.compat.v1.estimator.inputs.pandas_input_fn instead
    return input_fn

def training(classifier, train_input_fn):
    training_rmse = []
    validation_rmse = []

    steps = 300
    periods = 10
    steps_per_period = steps / periods

    for period in range(periods):
        classifier.train(input_fn=train_input_fn, steps=steps_per_period)


def main():
    event_df = get_input_data.get_events()
    df_train, df_test = data_preprocessing(event_df)

    fc_list = define_feature_columns(df_train)
    classifier = DNNBuilder(fc_list)

    train_input_fn = create_input_fn(df_train)
    test_input_fn = create_input_fn(df_test)

    training(classifier, train_input_fn)

    classifier.evaluate(input_fn=test_input_fn, steps=steps)



main()
