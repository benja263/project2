import numpy as np
import pickle
import os
from keras.layers import Dense, Dropout, Flatten, Convolution1D, MaxPooling1D,Embedding
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras
from src import transform_tweet
import tensorflow as tf
from keras.optimizers import TFOptimizer
import keras.optimizers
from src.preprocessing import pre_process_text
import random

dir = os.path.dirname(__file__)
VOCAB_PATH = os.path.join(dir, 'skipgram_dict.pickle')
# uncomment if cbow version
#VOCAB_PATH = os.path.join(dir, '..', 'fasttext_representation', 'cbow_dict.pickle')
POS_TRAIN_PATH = os.path.join(dir, '..', '..', 'data', 'raw', 'train_pos_full.txt')
NEG_TRAIN_PATH = os.path.join(dir, '..', '..',  'data', 'raw', 'train_neg_full.txt')
TEST_PATH = os.path.join(dir, '..', '..', 'data', 'preprocessed', 'parsed_test_data.txt')
SAVE_PATH = os.path.join(dir, 'final_ft_model_skipgram_dict.h5')

def main():
    print('Loading Vocabulary')
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)

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
    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(train_tweets)
    embedding_dim = 100
    embedding_matrix = transform_tweet.create_embedding_matrix(vocab, tokenizer.word_index, embedding_dim)
    padding_length = 40
    corpus = tokenizer.texts_to_sequences(train_tweets)
    # Shuffle
    c = list(zip(corpus, labels))
    random.shuffle(c)
    corpus,labels = zip(*c)
    """"""

    test_corpus = tokenizer.texts_to_sequences(test_tweets)

    test_corpus = pad_sequences(test_corpus, maxlen=padding_length, padding='post')
    corpus = pad_sequences(corpus, maxlen=padding_length, padding='post')
    labels = keras.utils.to_categorical(labels, 2)
    # saving test corpus
    with open('test_corpus_fasttext.pickle', 'wb') as handle:
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
              batch_size=128,
              shuffle=True)
    model.save(SAVE_PATH)
    with open('skipgram_final_fasttext_history.pickle', 'wb') as handle:
        pickle.dump(hist.history, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # uncomment if cbow version
    #with open('cbow_final_fasttext_history.pickle', 'wb') as handle:
    #    pickle.dump(hist.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
