import pandas as pd
from sklearn import neural_network
from sklearn.calibration import CalibratedClassifierCV
from sklearn.externals import joblib
import os

test_features = pd.read_csv("test_features.csv", index_col=0)
test_labels = pd.read_csv("test_labels.csv", index_col=0)
train_features = pd.read_csv("train_features.csv", index_col=0)
train_labels = pd.read_csv("train_labels.csv", index_col=0)

# Begin Training: Try to load the model and test data sets if the model already exists
if os.path.isfile('saved_model.pkl'):
    classifier = joblib.load('saved_model.pkl')
else:
    # Training
    classifier = neural_network.MLPClassifier(verbose=True, max_iter=200, hidden_layer_sizes=(10, 5))
    classifier = CalibratedClassifierCV(classifier, cv=3, method="isotonic")

    classifier.fit(train_features, train_labels)
    joblib.dump(classifier, 'saved_model.pkl')


# Testing: Place probabilities in a DF
test_predict_results_proba = classifier.predict_proba(test_features)
test_predict_results_proba = pd.DataFrame.from_records(test_predict_results_proba)

test_accuracy = classifier.score(test_features, test_labels)

print("Test Accuracy: ", test_accuracy)
print(test_predict_results_proba)

test_predict_results_proba.to_csv("predictions_sklearn.csv")

