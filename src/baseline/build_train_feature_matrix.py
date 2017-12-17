import numpy as np
import pickle
import os
import transform_tweet

dir = os.path.dirname(__file__)
POS_TRAIN_PATH = dir + '/../../data/raw/train_pos_full.txt'
NEG_TRAIN_PATH = dir +'/../../data/raw/train_neg_full.txt'
VOCAB_PATH = 'vocab.pkl'
EMBEDDING_PATH = 'embeddings.npy'
SAVE_FEATURE_PATH = 'features'
SAVE_LABEL_PATH = 'labels'

def main():
    print('Loading Vocabulary')
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)

    print('Loading Embedding Matrix')
    embedding_matrix = np.load(EMBEDDING_PATH)

    # Build training feature matrix:
    with open(POS_TRAIN_PATH) as f:
        pos_tweets = f.readlines()
    pos_feature_matrix = transform_tweet.old_tweetsToAvgVec(pos_tweets, vocab, embedding_matrix)

    with open(NEG_TRAIN_PATH) as f:
        neg_tweets = f.readlines()
    neg_feature_matrix = transform_tweet.old_tweetsToAvgVec(neg_tweets, vocab, embedding_matrix)

    full_feature_matrix = np.concatenate((pos_feature_matrix, neg_feature_matrix), axis=0)

    # Build label vector:
    pos_labels = np.ones(pos_feature_matrix.shape[0])
    neg_labels = -1*np.ones(neg_feature_matrix.shape[0])
    full_labels = np.concatenate((pos_labels, neg_labels), axis=0)

    # Shuffle
    shuffled_ind = np.arange(full_labels.shape[0])
    np.random.shuffle(shuffled_ind)

    full_feature_matrix = full_feature_matrix[shuffled_ind]
    full_labels = full_labels[shuffled_ind]

    # Save
    np.save(SAVE_FEATURE_PATH, full_feature_matrix)
    np.save(SAVE_LABEL_PATH, full_labels)

if __name__ == '__main__':
    main()