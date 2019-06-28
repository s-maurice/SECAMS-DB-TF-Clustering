import time
import numpy as np
import pandas as pd
import tensorflow as tf
import pyodbc
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as svm


def getEvents():
    #Connects to DB and grabs EVENT LOGS
    conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=DESKTOP-L3DV0JT;DATABASE=SECAMS;UID=sa;PWD=1')
    sqpQueryStr = """
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

    events = pd.read_sql(sqpQueryStr, conn)
    return events

def dayTimeNormed(events):
    decimalHour = (events.dt.hour + events.dt.minute / 60)

    tx = decimalHour.apply(lambda x: np.math.cos(x * np.pi /12))
    ty = decimalHour.apply(lambda x: np.math.sin(x * np.pi / 12))
    timexy = pd.DataFrame([tx, ty]).transpose()
    timexy.columns = ["daytx", "dayty"]
    return(timexy)

def dataPreprocessing(df):
    #In the case of SECAMS DB, all terminals are on thr same TERMINALGROUP, so drop as it may confuse the DNN
    df = df.drop(columns=["TERMINALGROUP", "TERMINALNAME"]) #TerminalName redundant as same as teminal id FUTURE USE MAY CONSIDER GROUPING BOYS/GIRLS VERSIONS OF SCHOOLS

    #Modify TIMESTAMPS to "day of week" "month of year" <- catagorical, and "normedtime x/y" as dense
    #normedtime x/y
    timexy = dayTimeNormed(df["TIMESTAMPS"])
    df = df.join(timexy)
    df['DAYTIME'] = df[['daytx', 'dayty']].values.tolist()
    #day of week
    dow = df["TIMESTAMPS"].dt.strftime("%a")
    df["DAYOFWEEK"] = dow
    #month of year
    moy = df["TIMESTAMPS"].dt.strftime("%b")
    df["MONTHOFYEAR"] = moy

    print(df[["TIMESTAMPS", "daytx", "dayty", "DAYOFWEEK", "MONTHOFYEAR"]])
    return df

    #remember to split into training and test datasets


def training_input_func(dataset):
    sparsedata = dataset.drop(["TIMESTAMPS", "daytx", "dayty"], axis=1) #Not nessacary, prep for iterable in future?

    #Create Feature Columns with each possible value for sparse data rows
    USERID_fc = tf.feature_column.categorical_column_with_vocabulary_list(key='USERID',vocabulary_list=sparsedata["USERID"].unique(), default_value=0)
    EVENTID_fc = tf.feature_column.categorical_column_with_vocabulary_list(key='EVENTID', vocabulary_list=sparsedata["EVENTID"].unique(), default_value=0)
    TERMINALSN_fc = tf.feature_column.categorical_column_with_vocabulary_list(key='TERMINALSN', vocabulary_list=sparsedata["TERMINALSN"].unique(), default_value=0)
    DOW_fc = tf.feature_column.categorical_column_with_vocabulary_list(key='DAYOFWEEK', vocabulary_list=sparsedata["DAYOFWEEK"].unique(), default_value=0)
    MOY_fc = tf.feature_column.categorical_column_with_vocabulary_list(key='MONTHOFYEAR', vocabulary_list=sparsedata["MONTHOFYEAR"].unique(), default_value=0)
    #numeric for timescale
    daytime_fc = tf.feature_column.numeric_column(key="DAYTIME", shape=[2,1]) #put both daytx and dayty in as array, double check the shape


    #Wrap within an embedding column
    #Sparse feature columns
    USERID_em = tf.feature_column.indicator_column(categorical_column=USERID_fc, dimension=3) #dimension should be about number_of_categories**0.25 according to google
    EVENTID_em = tf.feature_column.indicator_column(categorical_column=EVENTID_fc, dimension=2) #indicator or embedding column???
    TERMINALSN_em = tf.feature_column.indicator_column(categorical_column=TERMINALSN_fc, dimension=50)
    DOW_em = tf.feature_column.indicator_column(categorical_column=DOW_fc, dimension=1)
    MOY_em = tf.feature_column.indicator_column(categorical_column=MOY_fc, dimension=2)
    #dense feature columns - need to embed?

    #tf.estimator.DNNRegressor()

#CategoricalColumn

def main():
    eventDF = getEvents()
    eventDF = dataPreprocessing(eventDF)
    training_input_func(eventDF)
    print("ML Begin")




main()