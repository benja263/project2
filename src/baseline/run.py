import sklearn.model_selection
import sklearn.linear_model
import sklearn.neighbors
import sklearn.metrics
import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler


d = os.path.dirname(__file__)
TRAIN_FEATURE_PATH = 'features.npy'
TRAIN_LABEL_PATH = 'labels.npy'
TEST_FEATURE_PATH = 'test_features.npy'
TEST_IDS_PATH = os.path.join(d, '..', '..', 'data', 'preprocessed', 'ids.npy')

def main():
    # Load the data
    train_feature_matrix = np.load(TRAIN_FEATURE_PATH)
    train_label_vector = np.load(TRAIN_LABEL_PATH)

    test_feature_matrix = np.load(TEST_FEATURE_PATH)
    test_id_vector = np.load(TEST_IDS_PATH)

    # Scaling
    scaler = StandardScaler()
    scaler.fit(train_feature_matrix)
    train_feature_matrix = scaler.transform(train_feature_matrix)
    test_feature_matrix = scaler.transform(test_feature_matrix)

    # Define the classifier model
    clf = sklearn.linear_model.LogisticRegression(max_iter=100, tol=10e-10)
    #clf = sklearn.svm.SVC(max_iter=100, tol=10e-10)


    # Train model and use it to do prediction
    clf.fit(train_feature_matrix, train_label_vector)
    predictions = clf.predict(test_feature_matrix).astype(int)

    # Create the output csv file
    output_matrix = np.matrix([test_id_vector, predictions]).T
    df = pd.DataFrame(data=output_matrix)
    df.to_csv('submission.csv', header=['Id', 'Prediction'], sep=',', index=None)

if __name__ == '__main__':
    main()