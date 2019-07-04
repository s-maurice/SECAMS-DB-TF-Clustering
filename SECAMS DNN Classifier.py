import pandas as pd

import get_input_data


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


def train_model(split=[0.6, 0.2, 0.2]):
    raw_df = get_input_data.get_events()    # Get Raw DF
    df = raw_df.drop(columns=["TERMINALGROUP", "TERMINALNAME"]) # Drops unused columns
    df = df.dropna(how="any", axis=0) # Remove NANs
    df = df.sample(frac=1).reset_index(drop=True) #Shuffle Rows, reset index

    # Split data set into train, test, val
    split = [int(i / sum(split) * len(df)) for i in split]
    df_train = df.head(split[0])
    df_val = df.iloc[(split[0] + 1):(split[0] + split[1])]
    df_test = df.tail(split[2])





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