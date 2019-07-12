import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# A test classifier that attempts to find the 'Style' of each ramen product.

# The CSV filehas 2581 entries. It has the following headers:
# - Review #  (an integer)
# - Brand     (string)
# - Variety   (string)
# - Style     (string)
# - Country   (string)
# - Stars     (float)

# The model will attempt to use the following features:
# Brand   (categorical; 356 possible values)
# Country (categorical; 39 possible values)
# Stars   (numerical)
#
# The model will attempt to figure out the label:
# Style   (categorical; 5 possible values)

# More specifically, the possible targets (as 'Style') are:
# Pack / Bowl / Cup / Tray / Box
# However, only the first 4 seem to take on a significant number of values.

# The point of this classifier is to figure out whether the DNNClassifier we've set up can actually work (with larger sets of data).
# Since we're not going to be changing this much, a lot of things are going to be hard-coded (for the sake of convenience).


# First get the DF and split it
# raw_df = get_input_data.get_events()  # Get Raw DF
ramen_df = pd.read_csv("CSV Files/test_ramen/ramen-ratings.csv")


def split_df(df, split_array, shuffle=True):
    # Takes a DataFrame and splits it into 3 sets according to the ratio in the given array
    # split_array must have a length of 3.
    # shuffle: Shuffles the DataFrame before splitting

    assert(len(split_array) == 3)

    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    split = [int(i / sum(split_array) * len(df)) for i in split_array]

    df_head = df.head(split[0]).reset_index(drop=True)
    df_mid = df.iloc[(split[0] + 1):(split[0] + split[1])].reset_index(drop=True)
    df_tail = df.tail(split[2]).reset_index(drop=True)

    return [df_head, df_mid, df_tail]


def preprocess_features(df):
    processed_features = df.copy()[["Brand", "Country", "Stars"]]
    return processed_features


def preprocess_targets(df):
    processed_targets = pd.DataFrame()
    processed_targets["Style"] = df["Style"].apply(lambda x: str(x))
    return processed_targets


def construct_feature_columns():

    feature_column_list = []

    feature_column_list.append(tf.feature_column.numeric_column(key="Stars"))
    feature_column_list.append(tf.feature_column.indicator_column(categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(key="Brand", vocabulary_list=ramen_df["Brand"].unique())))
    feature_column_list.append(tf.feature_column.indicator_column(categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(key="Country", vocabulary_list=ramen_df["Country"].unique())))

    return feature_column_list


def create_input_function(features, targets, shuffle=True, batch_size=1, num_epochs=None):

    # Using tf.data (and DataSet)
    features = {key: np.array(value) for key, value in dict(features).items()}

    ds = tf.data.Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(features))
    feature_dict, label_list = ds.make_one_shot_iterator().get_next()

    return feature_dict, label_list


def train_model(
        train_features,
        train_targets,
        val_features,
        val_targets,
        learning_rate,
        batch_size,
        steps,
        hidden_units,
        periods=10,
        model_dir=None
):
    label_vocab_list = ramen_df["Style"].unique()
    label_vocab_list = [str(i) for i in label_vocab_list]

    # Create DNN
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate) # Create optimiser - Try variable rate optimisers
    classifier = tf.estimator.DNNClassifier(feature_columns=construct_feature_columns(),
                                            hidden_units=hidden_units,
                                            optimizer=optimizer,
                                            label_vocabulary=label_vocab_list,
                                            n_classes=len(label_vocab_list),
                                            model_dir=model_dir,
                                            config=tf.estimator.RunConfig().replace(save_summary_steps=10)) # Config bit is for tensorboard

    # Create input functions
    train_input_fn = lambda: create_input_function(train_features, train_targets, batch_size=batch_size)

    # ----- Begin Training + Train/Val Evaluation -----
    print("Training...")
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []

    # Train in periods; after every 'train', call .evaluate() and take accuracy
    steps_per_period = steps / periods

    for period in range(periods):
        classifier.train(input_fn=train_input_fn, steps=steps_per_period)

        eval_train_results = evaluate_model(classifier, train_features, train_targets, name="Training")
        eval_val_results = evaluate_model(classifier, val_features, val_targets, name="Validation")

        train_acc.append(eval_train_results.get('accuracy'))
        train_loss.append(eval_train_results.get('average_loss'))
        val_acc.append(eval_val_results.get('accuracy'))
        val_loss.append(eval_val_results.get('average_loss'))

        print("  Period %02d: Train: Accuracy = %f // Loss = %f // Average Loss = %f \n"
              "             Valid: Accuracy = %f // Loss = %f // Average Loss = %f" %
              (period, eval_train_results.get('accuracy'), eval_train_results.get('loss'),
               eval_train_results.get('average_loss'),
               eval_val_results.get('accuracy'), eval_val_results.get('loss'), eval_val_results.get('average_loss')))

    # All periods done
    print("Classifier trained.")

    # Graph the accuracy + average loss over the periods
    plt.subplot(121)
    plt.title("Accuracy vs. Periods (Learning rate: " + str(learning_rate) + ")")
    plt.xlabel("Periods")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)

    plt.plot(train_acc, label="training")
    plt.plot(val_acc, label="validation")
    plt.legend()

    plt.subplot(122)
    plt.title("Loss vs. Periods (Learning rate: " + str(learning_rate) + ")")
    plt.ylabel("Loss")
    plt.xlabel("Periods")

    plt.plot(train_loss, label="training")
    plt.plot(val_loss, label="validation")
    plt.legend()

    return classifier


def evaluate_model(model, features, targets, name=None):
    evaluate_input_function = lambda: create_input_function(features, targets, shuffle=False, num_epochs=1, batch_size=1)

    evaluate_result = model.evaluate(input_fn=evaluate_input_function, name=name)
    return evaluate_result


df_array = split_df(ramen_df, [8, 1, 1])  # Split into 3 DFs

# Assign train, validation and test features + targets
train_features = preprocess_features(df_array[0])
train_targets = preprocess_targets(df_array[0])

val_features = preprocess_features(df_array[1])
val_targets = preprocess_targets(df_array[1])

test_features = preprocess_features(df_array[2])
test_targets = preprocess_targets(df_array[2])

print(ramen_df.dtypes)

dnn_classifier = train_model(
        train_features,
        train_targets,
        val_features,
        val_targets,
        learning_rate=0.003,
        batch_size=20,
        steps=1500,
        hidden_units=[128, 64])

# --- MODEL TESTING ---

print("Classifier trained.")

# model.evaluate() on test results
eval_test_results = evaluate_model(dnn_classifier, test_features, test_targets, name="Test")
print("Test results:", eval_test_results)

plt.show()
