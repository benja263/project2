import numpy as np
import pickle
import os
import transform_tweet

# Use relative path to this file instead of relative path to the caller
dir = os.path.dirname(__file__)
POS_TRAIN_PATH = os.path.join(dir, '..', 'data', 'raw', 'train_pos.txt')
NEG_TRAIN_PATH = os.path.join(dir, '..', 'data', 'raw', 'train_neg.txt')
VOCAB_PATH = os.path.join(dir, '..', 'data', 'preprocessed', 'vocab.pkl')
EMBEDDING_PATH = os.path.join(dir, '..', 'data', 'embeddings.npy')
SAVE_FEATURE_PATH = os.path.join(dir, '..', 'data', 'features')
SAVE_LABEL_PATH = os.path.join(dir, '..',  'data', 'labels')

def main():
    print('Loading Vocabulary')
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)

    print('Loading Embedding Matrix')
    embedding_matrix = np.load(EMBEDDING_PATH)

    # Build training feature matrix:
    with open(POS_TRAIN_PATH) as f:
        pos_tweets = f.readlines()
    pos_feature_matrix = transform_tweet.tweetsToAvgVec(pos_tweets, vocab, embedding_matrix)

    with open(NEG_TRAIN_PATH) as f:
        neg_tweets = f.readlines()
    neg_feature_matrix = transform_tweet.tweetsToAvgVec(neg_tweets, vocab, embedding_matrix)

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



    """
    print('Loading Positive Training SET')
    N,K = np.shape(embedding_matrix)
    pos_training_matrix = np.zeros((10*N,K))
    row_counter = 0
    with open(POS_TRAIN_PATH) as tr_text_set:
        for tweet in tr_text_set:
            nb_words = len(tweet.split())
            words = np.zeros((nb_words,K))
            counter = 0
            zero_ind = np.zeros((nb_words))
            for word in tweet.split():
                index = vocab.get(word,'none')
                if isinstance(index,int):
                    words[counter] = embedding_matrix[index]
                    counter += 1
                else:
                    zero_ind[counter] = counter
            zero_ind = zero_ind[zero_ind != 0]
            if len(zero_ind):
                words = np.delete(words, zero_ind, axis=0)
            pos_training_matrix[row_counter] = np.sum(words,axis = 0)/np.shape(words)[0]
            row_counter += 1

    pos_training_matrix = np.delete(pos_training_matrix,np.where(pos_training_matrix == np.zeros((K))),axis = 0)
    y_pos = np.ones((pos_training_matrix.shape[0]))

    print('Loading Negative Training SET')
    neg_training_matrix = np.zeros((10*N,K))
    row_counter = 0
    with open(NEG_TRAIN_PATH) as tr_text_set:
        for tweet in tr_text_set:
            nb_words = len(tweet.split())
            words = np.zeros((nb_words,K))
            counter = 0
            zero_ind = np.zeros((nb_words))
            for word in tweet.split():
                index = vocab.get(word,'none')
                if isinstance(index,int):
                    words[counter] = embedding_matrix[index]
                    counter += 1
                else:
                    zero_ind[counter] = counter
            zero_ind = zero_ind[zero_ind != 0]
            if len(zero_ind):
                words = np.delete(words, zero_ind, axis=0)
            neg_training_matrix[row_counter] = np.sum(words,axis = 0)/np.shape(words)[0]
            row_counter += 1
    neg_training_matrix = np.delete(neg_training_matrix,np.where(neg_training_matrix == np.zeros((K))),axis = 0)
    y_neg = -np.ones((neg_training_matrix.shape[0]))

    feature_matrix = np.concatenate((pos_training_matrix,neg_training_matrix),axis = 0)
    labels = np.concatenate((y_pos,y_neg),axis = 0)
    np.save(SAVE_FEAURE_PATH, feature_matrix)
    np.save(SAVE_LABEL_PATH, labels)
    """


if __name__ == '__main__':
    main()