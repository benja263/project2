import pickle
import os
import numpy as np
from keras.layers import Dense,  Dropout, Flatten, Convolution1D, MaxPooling1D
from keras.models import Sequential
import keras
import pandas as pd

dir = os.path.dirname(__file__)
TRAIN_FEATURE_PATH = os.path.join(dir,'..','baseline','features.npy')
TRAIN_LABEL_PATH = os.path.join(dir,'..','baseline','labels.npy')
TEST_FEATURE_PATH = os.path.join(dir,'..','baseline','test_features.npy')
TEST_IDS_PATH = os.path.join(dir,'..','baseline','ids.npy')

def main():
    # Load the data
    train_feature_matrix = np.load(TRAIN_FEATURE_PATH)
    labels = np.load(TRAIN_LABEL_PATH)
    train_feature_matrix = train_feature_matrix[:,:,np.newaxis]

    test_feature_matrix = np.load(TEST_FEATURE_PATH)
    test_id_vector = np.load(TEST_IDS_PATH)
    test_feature_matrix = test_feature_matrix[:, :, np.newaxis]

    labels = keras.utils.to_categorical(labels, 2)
    # running neural nets
    print(train_feature_matrix.shape[1])
    model = Sequential()
    model.add(Convolution1D(64, 3, padding='valid', activation='relu', input_shape= (train_feature_matrix.shape[1],1)))
    #model.add(MaxPooling1D(strides=(2,)))
    model.add(Convolution1D(32, 3, padding='valid', activation='relu'))
    #model.add(MaxPooling1D(strides=(2,)))
    model.add(Convolution1D(16, 3, padding='valid', activation='relu'))
    #model.add(MaxPooling1D(strides=(2,)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(180, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    hist = model.fit(train_feature_matrix, labels,
                     epochs=3,
                     verbose=2,
                     validation_split=0.1,
                     batch_size=128,
                     shuffle=True)
    predictions = np.array(model.predict_classes(test_feature_matrix))
    predictions = np.squeeze(predictions)
    # Create the output csv file
    output_matrix = np.matrix([test_id_vector, predictions]).T
    df = pd.DataFrame(data=output_matrix)
    df.to_csv('submission_given_glove.csv', header=['Id', 'Prediction'], sep=',', index=None)

    with open('given_glove_history.pickle', 'wb') as handle:
        pickle.dump(hist.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()

