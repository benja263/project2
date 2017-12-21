import os
import numpy as np

d = os.path.dirname(__file__)
TEST_PATH = os.path.join('..', 'data', 'raw', 'test_data.txt')
SAVE_IDS_PATH = os.path.join('..', 'data', 'preprocessed', 'ids')
PARSED_TEST_DATA_PATH = os.path.join('..', 'data', 'preprocessed', 'parsed_test_data.txt')


def main():
    # Open the raw test dataset
    with open(TEST_PATH) as f:
        lines = f.readlines()

    # Create two lists, one for the ids and one for the tweets
    ids = [line.split(',', 1)[0] for line in lines]
    tweets = [line.split(',', 1)[1] for line in lines]

    # Save each list in a different file
    np.save(SAVE_IDS_PATH, ids)
    with open(PARSED_TEST_DATA_PATH, 'w') as f:
        for tweet in tweets:
            f.write(tweet)


if __name__ == '__main__':
    main()