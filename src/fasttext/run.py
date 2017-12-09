import fasttext
import os
import csv

dir = os.path.dirname(__file__)
TEST_PATH = os.path.join(dir, '..', '..', 'data', 'preprocessed', 'parsed_test_data.txt')

def main():
    with open(TEST_PATH) as f:
        lines = f.readlines()

    # Cross validation
    """
    classifier = fasttext.supervised('fasttext_train.txt', 'model')
    result = classifier.test('fasttext_validation.txt')
    print('P@1:', result.precision)
    print('R@1:', result.recall)
    print('Number of examples:', result.nexamples)
    """

    # Learning
    classifier = fasttext.supervised('fasttext_train.txt', 'model')

    # Prediction
    predictions = classifier.predict(lines)

    # Write submission file
    with open('submission.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['Id', 'Prediction'])
        for i in range(len(predictions)):
            csvwriter.writerow([i+1, int(predictions[i][0])])


if __name__ == '__main__':
    main()