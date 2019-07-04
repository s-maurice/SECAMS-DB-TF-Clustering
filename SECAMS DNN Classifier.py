import pandas as pd
import tensorflow as tf

import get_input_data


def split_df(df, split_array):
    # Takes a DataFrame and splits it into 3 sets according to the ratio in the given array
    # split_array must have a length of 3.

    assert(len(split_array == 3))

    split = [int(i / sum(split_array) * len(df)) for i in split_array]

    df_head = df.head(split[0])
    df_mid = df.iloc[(split[0] + 1):(split[0] + split[1])]
    df_tail = df.tail(split[2])

    return [df_head, df_mid, df_tail]


def preprocess_features(df):
    processed_features = pd.DataFrame()
    processed_features["DECHOUR"] = df["TIMESTAMPS"].apply(lambda x: x.dt.hour + x.dt.minute / 1440)  # Turns datetime format into decimalhour, normalised by day
    processed_features["DAYOFWEEK"] = df["TIMESTAMPS"].dt.strftime("%a")     # day of week
    processed_features["MONTHOFYEAR"] = df["TIMESTAMPS"].dt.strftime("%b")     # month of year
    return processed_features


def preprocess_targets(df):
    processed_targets = pd.DataFrame()
    processed_targets["USERID"] = df["USERID"]
    return processed_targets

def construct_feature_columns(numerical_columns_list, catagorical_columns_list, raw_df):

    numerical_features_list = []
    for i in numerical_columns_list:
        current_column = tf.feature_column.numeric_column(key=i)
        numerical_features_list.append(current_column)

    catagorical_features_list = []
    for i in catagorical_columns_list:
        current_column = tf.feature_column.categorical_column_with_vocabulary_list(key=i, vocabulary_list=raw_df[i].unique())
        # current_column = tf.feature_column.indicator_column(catagorical_column=current_column) # May need to wrap within indicator column
        catagorical_features_list.append(current_column)

    feature_column_list = numerical_features_list + catagorical_features_list
    return feature_column_list


def train_model(
        train_features,
        train_targets,
        val_features,
        val_targets,
        learning_rate = 0.001,
        batch_size = 1,
        steps_per_period = 50,
        periods = 10,
        hidden_units = [1024, 512, 256]
):

    raw_df = get_input_data.get_events()    # Get Raw DF
    df = raw_df.drop(columns=["TERMINALGROUP", "TERMINALNAME"]) # Drops unused columns
    df = df.dropna(how="any", axis=0) # Remove NANs
    df = df.sample(frac=1).reset_index(drop=True) #Shuffle Rows, reset index



    train_features = preprocess_features()
    val_features = preprocess_features()
    test_features = preprocess_features()

    train_targets = preprocess_targets()
    val_targets = preprocess_targets()
    test_targets = preprocess_targets()



    #TRAINING

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