import sklearn.model_selection
import sklearn.linear_model
import sklearn.neighbors
import sklearn.metrics
import os
import numpy as np
import pandas as pd

dir = os.path.dirname(__file__)
TRAIN_FEATURE_PATH = os.path.join(dir, '..', '..', 'data', 'features.npy')
TRAIN_LABEL_PATH = os.path.join(dir, '..', '..', 'data', 'labels.npy')
TEST_FEATURE_PATH = os.path.join(dir, '..', '..', 'data', 'test_features.npy')
TEST_IDS_PATH = os.path.join(dir, '..', '..', 'data', 'ids.npy')

def main():
    # Load the data
    train_feature_matrix = np.load(TRAIN_FEATURE_PATH)
    train_label_vector = np.load(TRAIN_LABEL_PATH)
    print(train_label_vector)

    test_feature_matrix = np.load(TEST_FEATURE_PATH)
    test_id_vector = np.load(TEST_IDS_PATH)

    # Define the classifier model
    clf = sklearn.linear_model.LogisticRegression(max_iter=1000, tol=10e-10)


    # Do 10-fold cross validation
    scoring = 'accuracy'
    kfold = sklearn.model_selection.KFold(n_splits=2, shuffle=True)
    cv_results = sklearn.model_selection.cross_val_score(clf, train_feature_matrix, train_label_vector, cv=kfold, scoring=scoring)
    msg = "%s: %f (%f)" % ('Logistic Regression', cv_results.mean(), cv_results.std())
    print(msg)

    # Train model and use it to do prediction
    clf.fit(train_feature_matrix, train_label_vector)
    predictions = clf.predict(test_feature_matrix).astype(int)

    # Create the output csv file
    output_matrix = np.matrix([test_id_vector, predictions]).T
    df = pd.DataFrame(data=output_matrix)
    df.to_csv('output.csv', header=['Id', 'Prediction'], sep=',', index=None)

if __name__ == '__main__':
    main()