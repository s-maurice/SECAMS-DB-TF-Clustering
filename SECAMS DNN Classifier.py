import math
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

import get_input_data


def split_df(df, split_array, shuffle=True):
    # Takes a DataFrame and splits it into 3 sets according to the ratio in the given array
    # split_array must have a length of 3.
    # shuffle: Shuffles the DataFrame before splitting

    assert(len(split_array) == 3)

    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    split = [int(i / sum(split_array) * len(df)) for i in split_array]

    df_head = df.head(split[0])
    df_mid = df.iloc[(split[0] + 1):(split[0] + split[1])]
    df_tail = df.tail(split[2])

    return [df_head, df_mid, df_tail]


def get_decimal_hour(events):
    decimal_hour = (events.dt.hour + events.dt.minute / 60)
    return decimal_hour


def preprocess_features(df):
    processed_features = pd.DataFrame()
    processed_features["DECHOUR"] = get_decimal_hour(df["TIMESTAMPS"]).apply(lambda x: x / 24)  # Turns datetime format into decimalhour, normalised by day
    processed_features["DAYOFWEEK"] = df["TIMESTAMPS"].dt.strftime("%w")     # day of week
    processed_features["MONTHOFYEAR"] = df["TIMESTAMPS"].dt.strftime("%-m")     # month of year
    processed_features["TERMINALSN"] = df["TERMINALSN"]
    processed_features["EVENTID"] = df["EVENTID"]

    # # debugging:
    # print(type(processed_features["DECHOUR"][0]))
    # print(type(processed_features["DAYOFWEEK"][0]))
    # print(type(processed_features["MONTHOFYEAR"][0]))
    # print(type(processed_features["TERMINALSN"][0]))
    # print(type(processed_features["EVENTID"][0]))
    #
    # print('processed features:\n', processed_features)

    return processed_features


def preprocess_targets(df):
    processed_targets = pd.DataFrame()
    processed_targets["USERID"] = df["USERID"].apply(lambda x: str(x))
    return processed_targets


def construct_feature_columns(numerical_columns_list, catagorical_columns_list, vocab_df):
    numerical_features_list = []
    for i in numerical_columns_list:
        current_column = tf.feature_column.numeric_column(key=i)
        numerical_features_list.append(current_column)

    categorical_features_list = []
    for i in catagorical_columns_list:
        current_column = tf.feature_column.categorical_column_with_vocabulary_list(key=i, vocabulary_list=vocab_df[i].unique())
        current_column = tf.feature_column.indicator_column(categorical_column=current_column) # May need to wrap within indicator column
        categorical_features_list.append(current_column)

    feature_column_list = numerical_features_list + categorical_features_list
    return feature_column_list


def create_input_function(features, targets, shuffle=True, batch_size=1, num_epochs=None):

    # DEPRECATED 1: Using tf.data (and DataSet)
    features = {key:np.array(value) for key,value in dict(features).items()}

    ds = tf.data.Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(features))
    feature_dict, label_list = ds.make_one_shot_iterator().get_next()

    # APPROACH 3: Directly turning it into a dict-list tuple
    # turn features DataFrame into Dict - input feature is a key, and then a list of values for the training batch
    # feature_dict = dict()
    #
    # for i in features.columns:
    #     feature_dict[str(i)] = features[i].tolist()
    #
    # # turn targets DataFrame into a List - these are our labels
    # label_list = targets[targets.columns[0]].tolist()

    return feature_dict, label_list


def train_model(
        train_features,
        train_targets,
        val_features,
        val_targets,
        learning_rate=0.001,
        batch_size=1,
        steps_per_period=50,
        periods=10,
        hidden_units=[1024, 512, 256],
        model_dir=None
):
    numerical_features = ["DECHOUR"]
    categorical_features = ["DAYOFWEEK", "MONTHOFYEAR", "TERMINALSN", "EVENTID"]

    # Create a vocab DataFrame by concatenating the given DFs.
    # NOTE: Should add test_features and test_targets to this later on as well.
    features_vocab_df = train_features.append(val_features)
    feature_columns = construct_feature_columns(numerical_features, categorical_features, features_vocab_df)

    # Prepare label_vocab
    label_vocab_list = train_targets["USERID"].unique()
    label_vocab_list = label_vocab_list.tolist()
    label_vocab_list = [str(i) for i in label_vocab_list]

    # Create DNN
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate) # Create optimiser - Try variable rate optimisers
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)

    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=hidden_units,
                                            optimizer=optimizer,
                                            label_vocabulary=label_vocab_list,
                                            n_classes=len(label_vocab_list),
                                            model_dir=model_dir,
                                            config=tf.estimator.RunConfig().replace(save_summary_steps=10)) # Config bit is for tensorboard

    # Create input functions
    train_input_fn = lambda: create_input_function(train_features, train_targets, batch_size=batch_size, num_epochs=10)

    # ----- Begin Training + Train/Val Evaluation -----
    print("Training...")
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []

    # Train in periods; after every 'train', call .evaluate() and take accuracy
    for period in range(periods):
        classifier.train(input_fn=train_input_fn, steps=steps_per_period)

        eval_train_results = evaluate_model(classifier, train_features, train_targets)
        eval_val_results = evaluate_model(classifier, val_features, val_targets)

        train_acc.append(eval_train_results.get('accuracy'))
        train_loss.append(eval_train_results.get('average_loss'))
        val_acc.append(eval_val_results.get('accuracy'))
        val_loss.append(eval_val_results.get('average_loss'))

        print("  Period %02d: Train: Accuracy = %f // Loss = %f // Average Loss = %f \n"
              "             Valid: Accuracy = %f // Loss = %f // Average Loss = %f" %
              (period, eval_train_results.get('accuracy'), eval_train_results.get('loss'), eval_train_results.get('average_loss'),
               eval_val_results.get('accuracy'), eval_val_results.get('loss'), eval_val_results.get('average_loss')))

    # All periods done
    print("Classifier trained.")

    # Graph the accuracy + average loss over the periods
    plt.subplot(311)
    plt.title("Accuracy vs. Periods (Learning rate: " + str(learning_rate) + ")")
    plt.xlabel("Periods")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)

    plt.plot(train_acc, label="training")
    plt.plot(val_acc, label="validation")
    plt.legend()

    plt.subplot(312)
    plt.title("Loss vs. Periods (Learning rate: " + str(learning_rate) + ")")
    plt.ylabel("Loss")
    plt.xlabel("Periods")

    plt.plot(train_loss, label="training")
    plt.plot(val_loss, label="validation")
    plt.legend()

    # Embedding Visualisation / Extraction

    # embeddings = tf.get_variable("embeddings", [len(targets), len(targets)])
    # print(embeddings)
    # embedded_word_ids = tf.nn.embedding_lookup(embeddings, features)
    # print(embedded_word_ids)
    #
    # print("Evaluation results:")
    # print(evaluate_result)

    # Currently working on  Below

    # sess = tf.Session()
    #
    # embedding = tf.Variable(tf.zeros([1024, embedding_size]), name="test_embedding")
    # assignment = embedding.assign(embedding_input)
    # saver = tf.train.Saver()
    #
    # sess.run(tf.global_variables_initializer())
    # writer = tf.summary.FileWriter(LOGDIR + hparam)
    # writer.add_graph(sess.graph)
    #
    # config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    # embedding_config = config.embeddings.add()
    # embedding_config.tensor_name = embedding.name
    # embedding_config.metadata_path = LABELS
    #
    # tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

    return classifier


# Function that tests a model against a set of features and targets;
# Verbose: Checks and prints the result of every single one
def evaluate_model(model, features, targets, name=None, steps=None):
    evaluate_input_function = lambda: create_input_function(features, targets, shuffle=False, num_epochs=1, batch_size=1)

    evaluate_result = model.evaluate(
        input_fn=evaluate_input_function,
        steps=steps,
        name=name)

    return evaluate_result


# function that directly predicts and compares all data points given, using a model
def predict_model(model, features, targets):

    predict_input_function = lambda: create_input_function(features, targets, shuffle=False, num_epochs=1, batch_size=1)

    predict_results = model.predict()

    for idx, prediction in enumerate(predict_results):
        print(idx, prediction)


def main():
    # raw_df = get_input_data.get_events()  # Get Raw DF
    raw_df = get_input_data.get_events_from_csv("SECAMS_common_user_id.csv")

    df_array = split_df(raw_df, [2, 2, 1])  # Split into 3 DFs

    # Assign train, validation and test features + targets
    train_features = preprocess_features(df_array[0])
    train_targets = preprocess_targets(df_array[0])

    val_features = preprocess_features(df_array[1])
    val_targets = preprocess_targets(df_array[1])

    test_features = preprocess_features(df_array[2])
    test_targets = preprocess_targets(df_array[2])

    dnn_classifier = train_model(
        train_features,
        train_targets,
        val_features,
        val_targets,
        learning_rate=0.0001,
        batch_size=500,
        steps_per_period=200,
        periods=10,
        model_dir="tmp/tf",
        hidden_units=[1024, 512, 256])

    plt.subplot(313)
    plt.title("UserID vs. Timestamps")
    plt.scatter(raw_df["TIMESTAMPS"], raw_df["USERID"])

    plt.show()
    # evaluate_model(dnn_classifier, train_features, train_targets, steps=600, verbose=False, name='Training')
    # evaluate_model(dnn_classifier, val_features, val_targets, steps=600, verbose=False, name='Validation')


main()
