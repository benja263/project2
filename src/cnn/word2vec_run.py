import pickle
import os
import numpy as np
from keras.layers import Dense, Dropout, Flatten,Convolution1D, MaxPooling1D, Embedding
from keras.models import Sequential
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from src import transform_tweet
from src.preprocessing import pre_process_text
import tensorflow as tf
from keras.optimizers import TFOptimizer
from gensim.models.keyedvectors import KeyedVectors
import random

dir = os.path.dirname(__file__)
POS_TRAIN_PATH = os.path.join(dir, '..', '..', 'data', 'raw', 'train_pos_full.txt')
NEG_TRAIN_PATH = os.path.join(dir, '..','..',  'data', 'raw', 'train_neg_full.txt')
TEST_PATH = os.path.join(dir, '..', '..', 'data', 'preprocessed', 'parsed_test_data.txt')
SAVE_PATH = os.path.join(dir,'final_word2vec_model.h5')

def main():

    # Build Training and Test matrices:
    print('Loading Positive Tweets')
    with open(POS_TRAIN_PATH) as f:
        pos_tweets = f.readlines()
    print('Loading Negative Tweets')
    with open(NEG_TRAIN_PATH) as f:
        neg_tweets = f.readlines()
    print('Load Test Tweets')
    with open(TEST_PATH) as f:
        test_tweets = f.readlines()
    pos_labels = np.ones((len(pos_tweets),1)).astype(int)
    neg_labels = np.zeros((len(neg_tweets),1)).astype(int)
    labels = np.squeeze(np.concatenate((pos_labels,neg_labels),axis=0))
    counter = 0
    for tweet in pos_tweets:
        tweet = pre_process_text.clean(tweet)
        pos_tweets[counter] = tweet
        counter += 1
    counter = 0
    for tweet in neg_tweets:
        tweet = pre_process_text.clean(tweet)
        neg_tweets[counter] = tweet
        counter += 1
    counter = 0
    for tweet in test_tweets:
        tweet = pre_process_text.clean(tweet)
        test_tweets[counter] = tweet
        counter += 1
    train_tweets = pos_tweets + neg_tweets
    corpus = []
    for row in [pos_tweets, neg_tweets]:
        for tweet in row:
            corpus.append(tweet.split())
    embedding_dim = 300
    # creating training and test matrices


    word_vector = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)  # C binary format
    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(train_tweets)
    padding_length = 40
    embedding_matrix = transform_tweet.create_word2Vec_embedding_matrix(word_vector, tokenizer.word_index, embedding_dim)
    corpus = tokenizer.texts_to_sequences(train_tweets)
    # Shuffle

    c = list(zip(corpus, labels))
    random.shuffle(c)
    corpus,labels = zip(*c)

    test_corpus = tokenizer.texts_to_sequences(test_tweets)

    test_corpus = pad_sequences(test_corpus, maxlen=padding_length, padding='post')
    corpus = pad_sequences(corpus, maxlen=padding_length, padding='post')
    labels = keras.utils.to_categorical(labels, 2)
    # saving test corpus
    with open('final_test_corpus_word2vec.pickle', 'wb') as handle:
        pickle.dump(test_corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # running neural nets
    model = Sequential()
    model.add(Embedding(len(tokenizer.word_index) + 1, embedding_dim, input_length=padding_length))
    model.layers[0].trainable = True
    model.layers[0].set_weights([embedding_matrix])
    model.add(Convolution1D(64, 3, padding='valid', activation = 'relu'))
    model.add(MaxPooling1D(strides=(2,)))
    model.add(Convolution1D(32, 3, padding='valid', activation = 'relu'))
    model.add(MaxPooling1D(strides=(2,)))
    model.add(Convolution1D(16, 3, padding='valid', activation = 'relu'))
    model.add(MaxPooling1D(strides=(2,)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='sigmoid'))
    model.summary()

    model.compile(loss='categorical_crossentropy',metrics=['accuracy'],
                  optimizer=TFOptimizer(tf.train.GradientDescentOptimizer(0.01)))
    hist = model.fit(corpus, labels,
              epochs=15,
              verbose=2,
              validation_split=0.1,
              shuffle=True)
    model.save(SAVE_PATH)
    with open('final_Word2vec_history.pickle', 'wb') as handle:
        pickle.dump(hist.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
