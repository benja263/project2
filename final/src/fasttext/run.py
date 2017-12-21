import fasttext
import csv
import os

d = os.path.dirname(__file__)
TRAIN_PATH = os.path.join(d, 'fasttext_train_final.txt')
TEST_PATH = os.path.join(d, '..', '..', 'data', 'preprocessed', 'parsed_test_data.txt')


def main():
    with open(TEST_PATH) as f:
        lines = f.readlines()

    # Learning
    classifier = fasttext.supervised(TRAIN_PATH, 'model', dim=100, epoch=5, word_ngrams=2, bucket=2000000)


    # Prediction
    predictions = classifier.predict(lines)


    # Write submission file
    with open('submission.csv', 'w') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['Id', 'Prediction'])
        for i in range(len(predictions)):
            csv_writer.writerow([i+1, int(predictions[i][0])])

if __name__ == '__main__':
    main()