import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

dir = os.path.dirname(__file__)
GLOVE_HISTORY_PATH = os.path.join(dir, 'final_glove_model_history.pickle')
FASTTEXT_HISTORY_PATH = os.path.join(dir, 'fast_final_textCNN_history.pickle')
WORD2VEC_HISTORY_PATH = os.path.join(dir, 'final_w2v_history.pickle')
def main():
    with open(GLOVE_HISTORY_PATH, 'rb') as f:
        glove_history = pickle.load(f)
    glove_val_loss = np.asarray(glove_history['val_loss'])
    glove_tr_loss = np.asarray(glove_history['loss'])
    glove_val_acc = np.asarray(glove_history['val_acc'])
    glove_tr_acc = np.asarray(glove_history['acc'])
    with open(FASTTEXT_HISTORY_PATH, 'rb') as f:
        fasttext_history = pickle.load(f)
    fasttext_val_loss = np.asarray(fasttext_history['val_loss'])
    fasttext_tr_loss = np.asarray(fasttext_history['loss'])
    fasttext_val_acc = np.asarray(fasttext_history['val_acc'])
    fasttext_tr_acc = np.asarray(fasttext_history['acc'])
    with open(WORD2VEC_HISTORY_PATH, 'rb') as f:
        w2v_history = pickle.load(f)
    w2v_val_loss = np.asarray(w2v_history['val_loss'])
    w2v_tr_loss = np.asarray(w2v_history['loss'])
    w2v_val_acc = np.asarray(w2v_history['val_acc'])
    w2v_tr_acc = np.asarray(w2v_history['acc'])

    epochs = np.linspace(1,20,20).astype(int)
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(epochs,glove_val_acc,color='b',marker='o',label='GloVe')
    plt.plot(epochs, fasttext_val_acc,color='r',marker='o',label='Fasttext')
    plt.plot(epochs, w2v_val_acc, color='g',marker='o',label='Word2Vec')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Prediction Percentage')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.subplot(2,2,2)
    plt.plot(epochs,glove_tr_acc,color='b',marker='o',label='GloVe')
    plt.plot(epochs, fasttext_tr_acc,color='r',marker='o',label='Fasttext')
    plt.plot(epochs, w2v_tr_acc, color='g',marker='o',label='Word2Vec')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Prediction Percentage')
    plt.title('Training Accuracy')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, glove_val_loss, color='b', marker='o', label='GloVe')
    plt.plot(epochs, fasttext_val_loss, color='r', marker='o', label='Fasttext')
    plt.plot(epochs, w2v_val_loss, color='g', marker='o', label='Word2Vec')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(epochs, glove_tr_loss, color='b', marker='o', label='GloVe')
    plt.plot(epochs, fasttext_tr_loss, color='r', marker='o', label='Fasttext')
    plt.plot(epochs, w2v_tr_loss, color='g', marker='o', label='Word2Vec')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


    print('glove validation accuaracy variance: {var}'.format(var =np.std(glove_val_acc)))
    print('fasttext validation accuaracy variance: {var}'.format(var=np.std(fasttext_val_acc)))
    print('word2vec validation accuaracy variance: {var}'.format(var=np.std(w2v_val_acc)))

    print('glove validation loss variance: {var}'.format(var=np.std(glove_val_loss)))
    print('fasttext validation loss variance: {var}'.format(var=np.std(fasttext_val_loss)))
    print('word2vec validation loss variance: {var}'.format(var=np.std(w2v_val_loss)))

    print('glove validation accuaracy mean: {var}'.format(var=np.mean(glove_val_acc)))
    print('fasttext validation accuaracy mean: {var}'.format(var=np.mean(fasttext_val_acc)))
    print('word2vec validation accuaracy mean: {var}'.format(var=np.mean(w2v_val_acc)))

    print('glove validation loss mean: {var}'.format(var=np.mean(glove_val_loss)))
    print('fasttext validation loss mean: {var}'.format(var=np.mean(fasttext_val_loss)))
    print('word2vec validation loss mean: {var}'.format(var=np.mean(w2v_val_loss)))



if __name__ == '__main__':
    main()