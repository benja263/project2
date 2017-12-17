import pickle
import os
import csv
import numpy as np
from gensim.models import Word2Vec
import keras
from src import transform_tweet

dir = os.path.dirname(__file__)
VOCAB_PATH = os.path.join(dir, '..', '..', 'data', 'preprocessed', 'vocab.pkl')
TEST_PATH = os.path.join(dir, '..', '..', 'data', 'preprocessed', 'parsed_test_data.txt')
VOCAB_PATH = os.path.join(dir,'word2vec.bin')
SAVE_PATH = os.path.join(dir,'w2v_model.h5')

def main():


    print('Loading Vocabulary')
    w2v = Word2Vec.load(VOCAB_PATH)
    # Build Test matrices:
    print('Loading Test Tweets ')
    with open(TEST_PATH) as f:
        test_tweets = f.readlines()
    test_corpus = []
    for tweet in test_tweets:
         test_corpus.append(tweet.split())
    embedding_dim = 200
    # creating training and test matrices
    test_matrix = transform_tweet.tweetsToAvgVec(test_corpus, w2v.wv, embedding_dim)
    # if convolutional nets where NOT used comment the line below!
    test_matrix = test_matrix[:, :, np.newaxis]
    # loading neural model
    model = keras.models.load_model(SAVE_PATH)

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    predictions = np.array(model.predict_classes(test_matrix))
    predictions[np.where(predictions == 0)] = -1
    predictions = np.squeeze(predictions)
    with open('submissionw2v.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['Id', 'Prediction'])
        for i in range(len(predictions)):
            csvwriter.writerow([i+1, int(predictions[i])])

if __name__ == '__main__':
    main()
