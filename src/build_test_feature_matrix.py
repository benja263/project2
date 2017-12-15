import numpy as np
import pandas as pd
import transform_tweet
import pickle
import os

# Use relative path to this file instead of relative path to the caller
dir = os.path.dirname(__file__)
TEST_PATH = os.path.join(dir, '..', 'data', 'preprocessed', 'parsed_test_data.txt')
VOCAB_PATH = os.path.join(dir, '..', 'data', 'preprocessed', 'vocab.pkl')
EMBEDDING_PATH = os.path.join(dir, '..', 'data', 'embeddings.npy')
SAVE_FEATURE_PATH = os.path.join(dir, '..', 'data', 'test_features')
#SAVE_IDS_PATH = os.path.join(dir, '..', 'data', 'ids')

def main():
    print('Loading Vocabulary')
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)

    print('Loading Embedding Matrix')
    embedding_matrix = np.load(EMBEDDING_PATH)

    with open(TEST_PATH) as f:
        tweets = f.readlines()

    feature_matrix = transform_tweet.tweetsToAvgVec(tweets, vocab, embedding_matrix)
    np.save(SAVE_FEATURE_PATH, feature_matrix)
    #np.save(SAVE_IDS_PATH, ids)


if __name__ == '__main__':
    main()