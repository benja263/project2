import pickle
import os
import numpy as np
from keras.layers import Dense, Embedding,  Dropout, Flatten, Convolution1D, MaxPooling1D
from keras.models import Sequential
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from src import transform_tweet
import re
from src.preprocessing import pre_process_text


dir = os.path.dirname(__file__)
VOCAB_PATH = os.path.join(dir,'glove_dict.pickle')
SAVE_LABEL_PATH = os.path.join(dir, '..', '..',  'data', '..', 'labels')
POS_TRAIN_PATH = os.path.join(dir, '..', '..', 'data', 'raw', 'train_pos.txt')
NEG_TRAIN_PATH = os.path.join(dir, '..','..',  'data', 'raw', 'train_neg.txt')
SAVE_PATH = os.path.join(dir,'glove_model.h5')
TEST_PATH = os.path.join(dir, '..', '..', 'data', 'preprocessed', 'parsed_test_data.txt')

# Google cloud version
#TEST_PATH = os.path.join(dir,'parsed_test_data.txt')  # google cloud version
#SAVE_LABEL_PATH = os.path.join(dir,'labels') # google cloud version
#POS_TRAIN_PATH = os.path.join(dir, 'train_pos_full.txt')  # google cloud version
#NEG_TRAIN_PATH = os.path.join(dir, 'train_neg_full.txt')  # google cloud version

def main():
    print('Loading Vocabulary')
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
    # Build Training and Test matrices:
    print('Loading Positive Tweets')
    with open(POS_TRAIN_PATH) as f:
        pos_tweets = f.readlines()
        '''
        print(lines)
        for line in lines:
            tweet = ""
            print(line)
            for word in line.split():
                print(word)
                #word = re.sub("[!?#\'<user><url>.]", "", word )
                word = re.sub("[!#.,]", "", word).replace("<user>", "")
                print(word)
            print(line)
        '''
    print('Loading Negative Tweets')
    with open(NEG_TRAIN_PATH) as f:
        neg_tweets = f.readlines()
    print('Load Test Tweets')
    with open(TEST_PATH) as f:
        test_tweets = f.readlines()
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
    embedding_dim = 300 # don't change unless create_glove_vocab was changed
    pos_labels = np.ones((len(pos_tweets),1)).astype(int)
    neg_labels = np.zeros((len(neg_tweets),1)).astype(int)
    labels = np.squeeze(np.concatenate((pos_labels,neg_labels),axis=0))
    corpus = []
    for row in [pos_tweets, neg_tweets]:
        for tweet in row:
            corpus.append(tweet.split())

    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(pos_tweets+neg_tweets)
    embedding_matrix = transform_tweet.create_embedding_matrix(vocab,tokenizer.word_index,embedding_dim)
    padding_length = 64
    corpus = tokenizer.texts_to_sequences(pos_tweets + neg_tweets)
    test_corpus = tokenizer.texts_to_sequences(test_tweets)

    test_corpus = pad_sequences(test_corpus, maxlen=padding_length, padding='post')
    corpus = pad_sequences(corpus, maxlen=padding_length, padding='post')
    labels = keras.utils.to_categorical(labels, 2)
    # saving test corpus
    with open('test_corpus_glove.pickle', 'wb') as handle:
        pickle.dump(test_corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)
    filter1 = 32
    filter2 = 16
    filter3 = 32
    filter4 = 16

    # running neural nets
    model = Sequential()
    #"""
    model.add(Embedding(len(tokenizer.word_index) + 1, embedding_dim, input_length=padding_length))
    model.layers[0].trainable = True
    model.layers[0].set_weights([embedding_matrix])
    """
    model.add(Convolution1D(64, filter1, padding='same', activation="relu"))
   # model.add(Convolution1D(32, filter2, padding='same', activation="relu"))
    model.add(MaxPooling1D(strides=(2,)))
    model.add(Convolution1D(16, filter3, padding='same', activation="relu"))
    #model.add(Convolution1D(8, filter4, padding='same', activation="relu"))
    model.add(MaxPooling1D(strides=(2,)))
    model.add(Flatten())
    model.add(Dense(256, activation='softmax'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))
    """
    model.add(Convolution1D(nb_filter=32, filter_length=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(2, activation='sigmoid'))

    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    model.fit(corpus, labels,
              epochs=3,
              verbose=2,
              validation_split=0.1,
              batch_size= 128,
              shuffle=True)
    model.save(SAVE_PATH)

if __name__ == '__main__':
    main()
