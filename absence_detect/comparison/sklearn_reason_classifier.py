import pandas as pd
from sklearn import neural_network
from sklearn.calibration import CalibratedClassifierCV
from sklearn.externals import joblib

test_features = pd.read_csv("test_features.csv")
test_labels = pd.read_csv("test_labels.csv")
train_features = pd.read_csv("train_features.csv")
train_labels = pd.read_csv("train_labels.csv")

# Training
classifier = neural_network.MLPClassifier(verbose=True, max_iter=200, hidden_layer_sizes=(100, 50))
classifier = CalibratedClassifierCV(classifier, cv=5, method="isotonic")
classifier.fit(train_features, train_labels)
joblib.dump(classifier, 'saved_model.pkl')


# Testing
test_predict_results_proba = classifier.predict_proba(test_features)
test_predict_results_proba_df = pd.DataFrame.from_records(test_predict_results_proba)
test_accuracy = classifier.score(test_features, test_labels)

print("Test Accuracy: ", test_accuracy)
print(test_predict_results_proba_df)

test_predict_results_proba_df.to_csv("sklearn_test_predict_results_proba_df")

