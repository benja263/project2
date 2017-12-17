import pickle
import os
import csv
import numpy as np
import keras


dir = os.path.dirname(__file__)
TEST_PATH = os.path.join(dir, 'test_corpus_fasttext.pickle')
SAVE_PATH = os.path.join(dir,'ft_model.h5')


def main():
    print('Loading Test Tweets ')
    with open(TEST_PATH, 'rb') as f:
        test_corpus = pickle.load(f)

    # loading neural model
    model = keras.models.load_model(SAVE_PATH)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    predictions = np.array(model.predict_classes(test_corpus))
    predictions[np.where(predictions == 0)] = -1
    predictions = np.squeeze(predictions)
    print(predictions)
    with open('submission_glove.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['Id', 'Prediction'])
        for i in range(len(predictions)):
            csvwriter.writerow([i + 1, int(predictions[i])])
if __name__ == '__main__':
    main()
