import pickle
import os
import numpy as np
import datetime
from gensim.models import Word2Vec
from keras.layers import Dense, Dropout, Flatten,Convolution1D, MaxPooling1D
from keras.models import Sequential
import keras
from src import transform_tweet

dir = os.path.dirname(__file__)
SAVE_LABEL_PATH = os.path.join(dir, '..', '..',  'data', '..', 'labels')
POS_TRAIN_PATH = os.path.join(dir, '..', '..', 'data', 'raw', 'train_pos.txt')
NEG_TRAIN_PATH = os.path.join(dir, '..','..',  'data', 'raw', 'train_neg.txt')
TEST_PATH = os.path.join(dir, '..', '..', 'data', 'preprocessed', 'parsed_test_data.txt')
SAVE_PATH = os.path.join(dir,'w2v_model.h5')

def main():

    # Build Training and Test matrices:
    print('Loading Positive Tweets')
    with open(POS_TRAIN_PATH) as f:
        pos_tweets = f.readlines()
    print('Loading Negative Tweets')
    with open(NEG_TRAIN_PATH) as f:
        neg_tweets = f.readlines()
    pos_labels = np.ones((len(pos_tweets),1)).astype(int)
    neg_labels = np.zeros((len(neg_tweets),1)).astype(int)
    labels = np.squeeze(np.concatenate((pos_labels,neg_labels),axis=0))
    corpus = []
    for row in [pos_tweets, neg_tweets]:
        for tweet in row:
            corpus.append(tweet.split())
    embedding_dim = 200
    # creating training and test matrices
    w2v = Word2Vec(corpus, size=embedding_dim,min_count=5,window=5)
    w2v.save('word2vec.bin')
    train_matrix = transform_tweet.tweetsToAvgVec(corpus, w2v.wv, embedding_dim)
    labels = keras.utils.to_categorical(labels, 2)

    filter1 = 3
    filter2 = 5
    filter3 = 3
    filter4 = 5

    # running convolutional neural nets
    #"""
    train_matrix = train_matrix[:, :, np.newaxis]
    model = Sequential()
    model.add(Convolution1D(64,filter1,input_shape=(train_matrix.shape[1],1), padding='same', activation="relu"))
    model.add(Convolution1D(32, filter2, padding='same', activation="relu"))
    model.add(MaxPooling1D(strides=(2,)))
    model.add(Convolution1D(16, filter3, padding='same', activation="relu"))
    model.add(Convolution1D(8, filter4, padding='same', activation="relu"))
    model.add(MaxPooling1D(strides=(2,)))
    model.add(Flatten())
    model.add(Dense(64, activation='softmax'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(2, activation='softmax'))
    #"""

    """ # normal neural net
    model.add(Dense(64, input_shape=(train_matrix.shape[1],), activation='softmax'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    """

    model.summary()

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    model.fit(train_matrix, labels,
              epochs=15,
              verbose=2,
              validation_split=0.1,
              shuffle=True)
    model.save(SAVE_PATH)


if __name__ == '__main__':
    main()
