import numpy as np
import transform_tweet
import pickle

# Use relative path to this file instead of relative path to the caller
TEST_PATH = 'parsed_test_data.txt'
VOCAB_PATH = 'vocab.pkl'
EMBEDDING_PATH = 'embeddings.npy'
SAVE_FEATURE_PATH = 'test_features'

def main():
    print('Loading Vocabulary')
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)

    print('Loading Embedding Matrix')
    embedding_matrix = np.load(EMBEDDING_PATH)

    with open(TEST_PATH) as f:
        tweets = f.readlines()

    feature_matrix = transform_tweet.old_tweetsToAvgVec(tweets, vocab, embedding_matrix)
    np.save(SAVE_FEATURE_PATH, feature_matrix)


if __name__ == '__main__':
    main()