import math
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
# import pyodbc
# import matplotlib.pyplot as plt
# import seaborn as sns
# import sklearn as svm

import get_input_data


def get_decimal_hour(events):
    decimal_hour = (events.dt.hour + events.dt.minute / 60)
    return decimal_hour


# DEPRECATED: Expresses time in a cyclical format through trigonometry
def day_time_normed(events):
    decimal_hour = get_decimal_hour(events)

    tx = decimal_hour.apply(lambda x: np.math.cos(x * np.pi / 12))
    ty = decimal_hour.apply(lambda x: np.math.sin(x * np.pi / 12))
    timexy = pd.DataFrame([tx, ty]).transpose()
    timexy.columns = ["daytx", "dayty"]

    return timexy


def data_preprocessing(df, split=[0.6, 0.2, 0.2]):
    # In the case of SECAMS DB, all terminals are on thr same TERMINALGROUP, so drop as it may confuse the DNN
    df = df.drop(columns=["TERMINALGROUP", "TERMINALNAME"])  # TerminalName redundant as same as teminal id FUTURE USE MAY CONSIDER GROUPING BOYS/GIRLS VERSIONS OF SCHOOLS

    # Drop NANs
    df = df.dropna(how="any", axis=0)

    # Modify TIMESTAMPS to "day of week" "month of year" <- categorical, and "normedtime x/y" as dense

    # # DEPRECATED: Using cyclical time, as an array with shape (2,1)
    # # normedtime x/y
    # timexy = day_time_normed(df["TIMESTAMPS"])
    # df = df.join(timexy)
    # df['DAYTIME'] = df[['daytx', 'dayty']].values.tolist()  # puts daytx and dayty into a single df column, with array dimension (2,1)

    # This section uses decimal hour time / 24
    df["DECHOUR"] = get_decimal_hour(df["TIMESTAMPS"]).apply(lambda x: x / 24)

    # day of week
    dow = df["TIMESTAMPS"].dt.strftime("%a")
    df["DAYOFWEEK"] = dow
    # month of year
    moy = df["TIMESTAMPS"].dt.strftime("%b")
    df["MONTHOFYEAR"] = moy

    # Shuffle DF Rows
    df = df.sample(frac=1).reset_index(drop=True)

    df = df.drop(["TIMESTAMPS"], axis=1) # Drop as unneeded and tf doesn't accept numpy datetime

    # Split data set into train, test, val
    # Multiply all values
    split = [int(i / sum(split) * len(df)) for i in split]

    df_train = df.head(split[0])
    df_val = df.iloc[(split[0] + 1):(split[0] + split[1])]
    df_test = df.tail(split[2])

    # df_train, df_test = train_test_split(df, test_size=0.2, random_state=1)  # Split into training and test data # Use stratify?
    # df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=1)  # Splits training into training and validation data
    return df_train, df_test, df_val


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


def dnn_builder(fc_list, learning_rate=0.001, hidden_units=[1024, 512, 256]):
    # Create an Optimiser
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)  # Experement with custom optmisers with variable rate
    # Build DNN Classifier - #USE DNNRegressor or DNNClassifier
    classifier = tf.estimator.DNNRegressor(feature_columns=fc_list, hidden_units=hidden_units, optimizer=optimizer)  # Not sure how many hidden units, layers/size, need more research/expermentation [1024, 512, 256]
    return classifier


def create_input_fn(df, shuffle=True, batch_size=1,num_epochs=0):
    # Places data into estimator
    input_fn = tf.estimator.inputs.pandas_input_fn(df, y=df["DECHOUR"], shuffle=shuffle, batch_size=batch_size, num_epochs=num_epochs)  # other params needed?, shuffle = true?, batch size # Deprecated, use tf.compat.v1.estimator.inputs.pandas_input_fn instead
    return input_fn


def training(classifier, train_input_fn, predict_train_input_fn, predict_val_input_fn, train_labels, val_labels, steps_per_period, periods):
    train_rmse = []
    val_rmse = []

    # print statement for RMSE values
    print("  period    | train   | val")

    for period in range(periods):
        # Train Model
        classifier.train(input_fn=train_input_fn, steps=steps_per_period)

        # Compute Predictions
        train_predictions = classifier.predict(input_fn=predict_train_input_fn)
        val_predictions = classifier.predict(input_fn=predict_val_input_fn)

        train_predictions_arr = np.array([item["predictions"][0] for item in train_predictions])
        val_predictions_arr = np.array([item["predictions"][0] for item in val_predictions])

        # Compute Loss
        train_rmse_current_tensor = metrics.mean_squared_error(train_labels, train_predictions_arr)
        val_rmse_current_tensor = metrics.mean_squared_error(val_labels, val_predictions_arr)

        train_rmse_current = math.sqrt(train_rmse_current_tensor)
        val_rmse_current = math.sqrt(val_rmse_current_tensor)

        # print(period, train_rmse_current, val_rmse_current)
        print("  period %02d : %0.6f, %0.6f" % (period, train_rmse_current, val_rmse_current))

        # Append RMSE to List
        train_rmse.append(train_rmse_current)
        val_rmse.append(val_rmse_current)

    return train_rmse, val_rmse

def testing(classifier, test_input_fn):
    test_input_fn = create_input_fn()


def rmse_plot(train, val):
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(train, label="training")
    plt.plot(val, label="validation")
    plt.axis([0, 10, 0, 0.2]) # Lock axis
    plt.legend()
    plt.show()


def main():
    # Hyper-parameters
    learning_rate = 0.0001              #
    batch_size = 200                    #
    steps_per_period = 100              #
    periods = 10                        #
    hidden_units = [1024, 512, 256]     # layers of DNN
    data_split_ratio = [6, 2, 2]        # ratio of data used for train, validation, testing (respectively)

    event_df = get_input_data.get_events()
    df_train, df_test, df_val = data_preprocessing(event_df, split=data_split_ratio)

    fc_list = define_feature_columns(df_train)
    classifier = dnn_builder(fc_list, learning_rate=learning_rate, hidden_units=hidden_units)

    # Create input functions; note create_input_fn automatically sets batch_size=1, shuffle=true, and num_epochs=0
    # For training - NOTE: num_epochs=0 doesn't seem to work; find a way to set num_epochs to 'None'.
    train_input_fn = create_input_fn(df_train, batch_size=batch_size, num_epochs=10)

    # For finding RMSE values
    predict_train_input_fn = create_input_fn(df_train, shuffle=False, num_epochs=1)
    predict_val_input_fn = create_input_fn(df_val, shuffle=False, num_epochs=1)
    predict_test_input_fn = create_input_fn(df_test, shuffle=False, num_epochs=1)

    # Training and Validation, plotting RMSE
    train_rmse, val_rmse = training(classifier, train_input_fn, predict_train_input_fn, predict_val_input_fn, df_train["DECHOUR"], df_val["DECHOUR"], steps_per_period=steps_per_period, periods=periods)
    rmse_plot(train_rmse, val_rmse)

    # classifier.evaluate(input_fn=test_input_fn, steps=300)


main()
