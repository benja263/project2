import numpy as np
import pickle
import os

# Use relative path to this file instead of relative path to the caller
dir = os.path.dirname(__file__)
TEST_PATH = os.path.join(dir, '..', 'data', 'raw', 'test_data.txt')
VOCAB_PATH = os.path.join(dir, '..', 'data', 'preprocessed', 'test_vocab.pkl')
EMBEDDING_PATH = os.path.join(dir, '..', 'data', 'test_embeddings.npy')
SAVE_FEAURE_PATH = os.path.join(dir, '..', 'data', 'test_features')
SAVE_LABEL_PATH = os.path.join(dir, '..',  'data', 'ids')

def main():
    print('Loading Vocabulary')
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
    print(len(vocab))
    print('Loading Embedding Matrix')
    embedding_matrix = np.load(EMBEDDING_PATH)
    print(embedding_matrix.shape)
    print('Loading TEST SET')
    N,K = np.shape(embedding_matrix)
    test_matrix = np.zeros((10*N,K))
    row_counter = 0
    ids = np.zeros((10*N))
    id_counter = 0
    with open(TEST_PATH) as tr_text_set:
        for tweet in tr_text_set:
            print(tweet)
            phrase = tweet.split(",")
            ids[id_counter] = int(phrase[0])
            id_counter += 1
            nb_words = len(phrase[1].split())
            words = np.zeros((nb_words,K))
            counter = 0
            zero_ind = np.zeros((nb_words))
            for word in phrase[1].split():
                index = vocab.get(word,'none')
                if isinstance(index,int):
                    words[counter] = embedding_matrix[index]
                    counter += 1
                else:
                    zero_ind[counter] = counter
            zero_ind = zero_ind[zero_ind != 0]
            if len(zero_ind):
                words = np.delete(words, zero_ind, axis=0)
            test_matrix[row_counter] = np.sum(words,axis = 0)/np.shape(words)[0]
            row_counter += 1
    test_matrix = np.delete(test_matrix,np.where(test_matrix == np.zeros((K))),axis = 0)
    ids =np.delete(ids,np.where(ids == 0))
    np.save(SAVE_FEAURE_PATH, test_matrix)
    np.save(SAVE_LABEL_PATH, ids)





if __name__ == '__main__':
    main()