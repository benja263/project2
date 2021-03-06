import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import matplotlib.ticker as ticker

dir = os.path.dirname(__file__)
GLOVE_HISTORY_PATH = os.path.join(dir, 'final_glove_history.pickle')
FASTTEXT_CBOW_HISTORY_PATH = os.path.join(dir, 'final_ft_history_cbow.pickle')
FASTTEXT_SKIPGRAM_HISTORY_PATH = os.path.join(dir, 'final_ft_history_skipgram.pickle')
WORD2VEC_HISTORY_PATH = os.path.join(dir, 'final_Word2vec_history.pickle')
def main():
    with open(GLOVE_HISTORY_PATH, 'rb') as f:
        glove_history = pickle.load(f)
    glove_val_loss = np.asarray(glove_history['val_loss'])
    glove_tr_loss = np.asarray(glove_history['loss'])
    glove_val_acc = np.asarray(glove_history['val_acc'])
    glove_tr_acc = np.asarray(glove_history['acc'])
    with open(FASTTEXT_CBOW_HISTORY_PATH, 'rb') as f:
        fasttext_history = pickle.load(f)
    ft_cbow_val_loss = np.asarray(fasttext_history['val_loss'])
    ft_cbow_tr_loss = np.asarray(fasttext_history['loss'])
    ft_cbow_val_acc = np.asarray(fasttext_history['val_acc'])
    ft_cbow_tr_acc = np.asarray(fasttext_history['acc'])
    with open(FASTTEXT_SKIPGRAM_HISTORY_PATH, 'rb') as f:
        fasttext_history = pickle.load(f)
    ft_skipgram_val_loss = np.asarray(fasttext_history['val_loss'])
    ft_skipgram_tr_loss = np.asarray(fasttext_history['loss'])
    ft_skipgram_val_acc = np.asarray(fasttext_history['val_acc'])
    ft_skipgram_tr_acc = np.asarray(fasttext_history['acc'])
    with open(WORD2VEC_HISTORY_PATH, 'rb') as f:
        w2v_history = pickle.load(f)
    w2v_val_loss = np.asarray(w2v_history['val_loss'])
    w2v_tr_loss = np.asarray(w2v_history['loss'])
    w2v_val_acc = np.asarray(w2v_history['val_acc'])
    w2v_tr_acc = np.asarray(w2v_history['acc'])

    epochs = np.linspace(1,15,15).astype(int)

    fig = plt.figure()
    plt.subplot(2,2,1)
    plt.plot(epochs,glove_val_acc,color='b',marker='o',label='GloVe')
    plt.plot(epochs, ft_cbow_val_acc,color='r',marker='o',label='Fasttext CBOW')
    plt.plot(epochs, ft_skipgram_val_acc, color='m', marker='o', label='Fasttext SKIPGRAM')
    plt.plot(epochs, w2v_val_acc, color='g',marker='o',label='Word2Vec')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Prediction Percentage')
    plt.title('Validation Accuracy')
    plt.legend()
    ax = fig.gca()
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    plt.subplot(2,2,2)
    plt.plot(epochs,glove_tr_acc,color='b',marker='o',label='GloVe')
    plt.plot(epochs, ft_cbow_tr_acc,color='r',marker='o',label='Fasttext CBOW')
    plt.plot(epochs, ft_skipgram_tr_acc, color='m', marker='o', label='Fasttext SKIPGRAM')
    plt.plot(epochs, w2v_tr_acc, color='g',marker='o',label='Word2Vec')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Prediction Percentage')
    plt.title('Training Accuracy')
    plt.legend()
    ax = fig.gca()
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    plt.subplot(2, 2, 3)
    plt.plot(epochs, glove_val_loss, color='b', marker='o', label='GloVe')
    plt.plot(epochs, ft_cbow_val_loss,color='r',marker='o',label='Fasttext CBOW')
    plt.plot(epochs, ft_skipgram_val_loss, color='m', marker='o', label='Fasttext SKIPGRAM')
    plt.plot(epochs, w2v_val_loss, color='g', marker='o', label='Word2Vec')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()
    ax = fig.gca()
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    plt.subplot(2, 2, 4)
    plt.plot(epochs, glove_tr_loss, color='b', marker='o', label='GloVe')
    plt.plot(epochs, ft_cbow_tr_loss,color='r',marker='o',label='Fasttext CBOW')
    plt.plot(epochs, ft_skipgram_tr_loss, color='m', marker='o', label='Fasttext SKIPGRAM')
    plt.plot(epochs, w2v_tr_loss, color='g', marker='o', label='Word2Vec')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    ax = fig.gca()
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()