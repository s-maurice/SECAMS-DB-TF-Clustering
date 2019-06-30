import time
import numpy as np
import pandas as pd
import tensorflow as tf
import pyodbc
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as svm


def getEvents():
    # Connects to DB and grabs EVENT LOGS
    conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=DESKTOP-L3DV0JT;DATABASE=SECAMS;UID=sa;PWD=1')
    sql_query_str = """
SELECT TOP(1000) USERID, EVENTID, TIMESTAMPS, access_event_logs.TERMINALSN, TERMINALGROUP, TERMINALNAME
FROM access_event_logs, access_terminal
WHERE 1 = CASE
WHEN ISNUMERIC(USERID) = 1 THEN CASE WHEN CAST(USERID as BIGINT) < 1000000 THEN 1 END
END
AND 
TIMESTAMPS > '2003-1-1'
AND 
access_event_logs.TerminalSN = access_terminal.TerminalSN
ORDER BY USERID"""

    events = pd.read_sql(sql_query_str, conn)
    return events


def day_time_normed(events):
    decimal_hour = (events.dt.hour + events.dt.minute / 60)

    tx = decimal_hour.apply(lambda x: np.math.cos(x * np.pi / 12))
    ty = decimal_hour.apply(lambda x: np.math.sin(x * np.pi / 12))
    timexy = pd.DataFrame([tx, ty]).transpose()
    timexy.columns = ["daytx", "dayty"]

    return timexy


def dataPreprocessing(df):
    # In the case of SECAMS DB, all terminals are on thr same TERMINALGROUP, so drop as it may confuse the DNN
    df = df.drop(columns=["TERMINALGROUP",
                          "TERMINALNAME"])  # TerminalName redundant as same as teminal id FUTURE USE MAY CONSIDER GROUPING BOYS/GIRLS VERSIONS OF SCHOOLS

    # Modify TIMESTAMPS to "day of week" "month of year" <- catagorical, and "normedtime x/y" as dense
    # normedtime x/y
    timexy = day_time_normed(df["TIMESTAMPS"])
    df = df.join(timexy)
    df['DAYTIME'] = df[
        ['daytx', 'dayty']].values.tolist()  # puts daytx and dayty into a single df column, with array dimension (2,1)
    # day of week
    dow = df["TIMESTAMPS"].dt.strftime("%a")
    df["DAYOFWEEK"] = dow
    # month of year
    moy = df["TIMESTAMPS"].dt.strftime("%b")
    df["MONTHOFYEAR"] = moy

    # Shuffle DF Rows
    df = df.sample(frac=1).reset_index(drop=True)  # could use shuffle(df) from sklearn.utils
    df_size = len(df.index)
    # Select ratio of train to test data - NEEDS REMAKE to handle decimal, give train_test_ratio differently - TEMP
    train_test_ratio = (4, 5)

    train_len = int(df_size * train_test_ratio[0] / train_test_ratio[1])
    test_len = int(df_size - (df_size * train_test_ratio[0] / train_test_ratio[1]))

    df_train = df.head(train_len)
    df_test = df.tail(test_len)

    return df_train, df_test


def define_feature_columns(dataset):
    sparse_df = dataset.drop(["TIMESTAMPS", "daytx", "dayty"], axis=1)  # Not nessacary, prep for iterable in future?

    # Create Feature Columns with each possible value for sparse data rows
    USERID_fc = tf.feature_column.categorical_column_with_vocabulary_list(key='USERID', vocabulary_list=sparse_df["USERID"].unique(), default_value=0)
    EVENTID_fc = tf.feature_column.categorical_column_with_vocabulary_list(key='EVENTID', vocabulary_list=sparse_df["EVENTID"].unique(), default_value=0)
    TERMINALSN_fc = tf.feature_column.categorical_column_with_vocabulary_list(key='TERMINALSN', vocabulary_list=sparse_df["TERMINALSN"].unique(), default_value=0)
    DOW_fc = tf.feature_column.categorical_column_with_vocabulary_list(key='DAYOFWEEK', vocabulary_list=sparse_df["DAYOFWEEK"].unique(), default_value=0)
    MOY_fc = tf.feature_column.categorical_column_with_vocabulary_list(key='MONTHOFYEAR', vocabulary_list=sparse_df["MONTHOFYEAR"].unique(), default_value=0)
    # numeric for timescale
    daytime_fc = tf.feature_column.numeric_column(key="DAYTIME", shape=[2, 1])  # put both daytx and dayty in as array, double check the shape

    feature_columns_list = [USERID_fc, EVENTID_fc, TERMINALSN_fc, DOW_fc, MOY_fc, daytime_fc]

    # Wrap within an embedding column
    # Sparse feature columns
    # USERID_em = tf.feature_column.indicator_column(categorical_column=USERID_fc) # dimension should be about number_of_categories**0.25 according to google, for embedding columns
    # EVENTID_em = tf.feature_column.indicator_column(categorical_column=EVENTID_fc) #indicator or embedding column???
    # TERMINALSN_em = tf.feature_column.indicator_column(categorical_column=TERMINALSN_fc)
    # DOW_em = tf.feature_column.indicator_column(categorical_column=DOW_fc)
    # MOY_em = tf.feature_column.indicator_column(categorical_column=MOY_fc)
    # #dense feature columns - need to embed?

    return feature_columns_list


def DNNBuilder(fc_list):
    # Build DNN Classifier - #USE DNNRegressor or DNNClassifier
    classifier = tf.estimator.DNNClassifier(feature_columns=fc_list, hidden_units=[256, 32])  # Not sure how many hidden units, layers/size, need more research/expermentation
    return classifier


def train_input_fn(df):
    # Places data into estimator
    input_fn = tf.estimator.inputs.pandas_input_fn(df, shuffle=True)  # other params needed?
    return input_fn


# CategoricalColumn

def main():
    event_df = getEvents()
    df_train, df_test = dataPreprocessing(event_df)
    fc_list = define_feature_columns(df_train)
    print("DNNBuilder")
    classifier = DNNBuilder(fc_list)
    train_input_fn(df_train)


main()
