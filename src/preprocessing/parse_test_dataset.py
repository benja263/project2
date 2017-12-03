import os
import numpy as np

dir = os.path.dirname(__file__)
TEST_PATH = os.path.join(dir, '..', '..', 'data', 'raw', 'test_data.txt')
SAVE_IDS_PATH = os.path.join(dir, '..', '..', 'data', 'ids')
PARSED_TEST_DATA_PATH = os.path.join(dir, '..', '..', 'data', 'preprocessed', 'parsed_test_data.txt')

def main():
    with open(TEST_PATH) as f:
        lines = f.readlines()

    ids = [line.split(',', 1)[0] for line in lines]
    tweets = [line.split(',', 1)[1] for line in lines]

    np.save(SAVE_IDS_PATH, ids)

    with open(PARSED_TEST_DATA_PATH, 'w') as f:
        for tweet in tweets:
            f.write(tweet)

if __name__ == '__main__':
    main()