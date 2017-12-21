import sklearn.model_selection
import sklearn.linear_model
import sklearn.neighbors
import sklearn.metrics
import numpy as np
import pandas as pd

TRAIN_FEATURE_PATH = 'features.npy'
TRAIN_LABEL_PATH = 'labels.npy'

def main():
    # Load the data
    train_feature_matrix = np.load(TRAIN_FEATURE_PATH)
    train_label_vector = np.load(TRAIN_LABEL_PATH)

    # Define the classifier model
    clf = sklearn.linear_model.LogisticRegression(max_iter=100, tol=10e-10)

    # Do 10-fold cross validation
    scoring = 'accuracy'
    kfold = sklearn.model_selection.KFold(n_splits=2, shuffle=True)
    cv_results = sklearn.model_selection.cross_val_score(clf, train_feature_matrix, train_label_vector, cv=kfold, scoring=scoring)
    msg = "%s: %f (%f)" % ('Logistic Regression', cv_results.mean(), cv_results.std())
    print(msg)

if __name__ == '__main__':
    main()