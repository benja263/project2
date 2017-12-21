import matplotlib.pyplot as plt
import fasttext
import os

d = os.path.dirname(__file__)
TRAIN_PATH = os.path.join(d, 'text', 'fasttext_train.txt')
VALIDATION_PATH = os.path.join(d, 'text', 'fasttext_validation.txt')

FILTERED_TRAIN_PATH = os.path.join(d, 'text', 'filtered_fasttext_train.txt')
FILTERED_VALIDATION_PATH = os.path.join(d, 'text', 'filtered_fasttext_validation.txt')

SW_TRAIN_PATH = os.path.join(d, 'text', 'sw_fasttext_train.txt')
SW_VALIDATION_PATH = os.path.join(d, 'text', 'sw_fasttext_validation.txt')

def get_score(model, validation_path):
    """
    Returns the score of a model after testing in on a validation set

    :param model: The fastText model
    :param validation_path: The path to the validation set file
    :return: The prediction score on the validation set
    """
    result = model.test(validation_path)
    return result.precision

def plot_dim():
    dims = [10, 20, 30, 40, 50, 100, 200, 300]
    score = []
    filtered_score = []
    sw_score = []

    for dim in dims:
        clf = fasttext.supervised(TRAIN_PATH, 'model', lr=0.05, dim=dim, ws=5, epoch=5,
                                  word_ngrams=1, bucket=2000000, loss='softmax', minn=3, maxn=6)
        filtered_clf = fasttext.supervised(FILTERED_TRAIN_PATH, 'model_filtered', lr=0.05, dim=dim, ws=5, epoch=5,
                                           word_ngrams=1, bucket=2000000, loss='softmax', minn=3, maxn=6)
        sw_clf = fasttext.supervised(SW_TRAIN_PATH, 'model_sw', lr=0.05, dim=dim, ws=5, epoch=5,
                                     word_ngrams=1, bucket=2000000, loss='softmax', minn=3, maxn=6)

        score.append(get_score(clf, VALIDATION_PATH))
        filtered_score.append(get_score(filtered_clf, FILTERED_VALIDATION_PATH))
        sw_score.append(get_score(sw_clf, SW_VALIDATION_PATH))

    plt.plot(dims, score, 'r', label='no filtering')
    plt.plot(dims, filtered_score, 'g', label='filtered')
    plt.plot(dims, sw_score, 'b', label='filtered + sw')
    plt.title("Validation score as a function of word vector dimensionality")
    plt.xlabel("Word vector dimensionality")
    plt.ylabel("Validation score")
    plt.legend()
    plt.axis([min(dims), max(dims), 0, 1])
    plt.show()

def plot_ws():
    wss = [5, 10, 15, 20, 25]
    score = []
    filtered_score = []
    sw_score = []
    for ws in wss:
        clf = fasttext.supervised(TRAIN_PATH, 'model', lr=0.05, dim=100, ws=ws, epoch=5,
                                  word_ngrams=1, bucket=2000000, loss='softmax', minn=3, maxn=6)
        filtered_clf = fasttext.supervised(FILTERED_TRAIN_PATH, 'model_filtered', lr=0.05, dim=100, ws=ws, epoch=5,
                                           word_ngrams=1, bucket=2000000, loss='softmax', minn=3, maxn=6)
        sw_clf = fasttext.supervised(SW_TRAIN_PATH, 'model_sw', lr=0.05, dim=100, ws=ws, epoch=5,
                                     word_ngrams=1, bucket=2000000, loss='softmax', minn=3, maxn=6)

        score.append(get_score(clf, VALIDATION_PATH))
        filtered_score.append(get_score(filtered_clf, FILTERED_VALIDATION_PATH))
        sw_score.append(get_score(sw_clf, SW_VALIDATION_PATH))

    plt.plot(wss, score, 'r', label='no filtering')
    plt.plot(wss, filtered_score, 'g', label='filtered')
    plt.plot(wss, sw_score, 'b', label='filtered + sw')
    plt.title("Validation score as a function of window size")
    plt.xlabel("Window size (in word count)")
    plt.ylabel("Validation score")
    plt.legend()
    plt.axis([min(wss), max(wss), 0, 1])
    plt.show()

def plot_epoch():
    epochs = [5, 10, 15, 20]
    score = []
    filtered_score = []
    sw_score = []
    for epoch in epochs:
        clf = fasttext.supervised(TRAIN_PATH, 'model', lr=0.05, dim=100, ws=5, epoch=epoch,
                                  word_ngrams=1,bucket=2000000, loss='softmax', minn=3,maxn=6)
        filtered_clf = fasttext.supervised(FILTERED_TRAIN_PATH, 'model_filtered', lr=0.05, dim=100, ws=5, epoch=epoch,
                                           word_ngrams=1,bucket=2000000, loss='softmax', minn=3,maxn=6)
        sw_clf = fasttext.supervised(SW_TRAIN_PATH, 'model_sw', lr=0.05, dim=100, ws=5, epoch=epoch,
                                           word_ngrams=1, bucket=2000000, loss='softmax', minn=3, maxn=6)

        score.append(get_score(clf, VALIDATION_PATH))
        filtered_score.append(get_score(filtered_clf, FILTERED_VALIDATION_PATH))
        sw_score.append(get_score(sw_clf, SW_VALIDATION_PATH))

    plt.plot(epochs, score, 'r', label='no filtering')
    plt.plot(epochs, filtered_score, 'g', label='filtered')
    plt.plot(epochs, sw_score, 'b', label='filtered + sw')
    plt.title("Validation score as a function of epoch number")
    plt.xlabel("Epoch number")
    plt.ylabel("Validation score")
    plt.legend()
    plt.axis([min(epochs), max(epochs), 0, 1])
    plt.show()

def plot_word_ngrams():
    word_ngrams = [1, 2, 3, 4, 5]
    score = []
    filtered_score = []
    sw_score = []
    for wng in word_ngrams:
        clf = fasttext.supervised(TRAIN_PATH, 'model', lr=0.05, dim=100, ws=5, epoch=5,
                                  word_ngrams=wng, bucket=2000000, loss='softmax', minn=3, maxn=6)
        filtered_clf = fasttext.supervised(FILTERED_TRAIN_PATH, 'model_filtered', lr=0.05, dim=100, ws=5, epoch=5,
                                           word_ngrams=wng, bucket=2000000, loss='softmax', minn=3, maxn=6)
        sw_clf = fasttext.supervised(SW_TRAIN_PATH, 'model_sw', lr=0.05, dim=100, ws=5, epoch=5,
                                     word_ngrams=wng, bucket=2000000, loss='softmax', minn=3, maxn=6)

        score.append(get_score(clf, VALIDATION_PATH))
        filtered_score.append(get_score(filtered_clf, FILTERED_VALIDATION_PATH))
        sw_score.append(get_score(sw_clf, SW_VALIDATION_PATH))

    plt.plot(word_ngrams, score, 'r', label='no filtering')
    plt.plot(word_ngrams, filtered_score, 'g', label='filtered')
    plt.plot(word_ngrams, sw_score, 'b', label='filtered + sw')
    plt.title("Validation score as a function of word n-grams length")
    plt.xlabel("Length of word n-grams")
    plt.ylabel("Validation score")
    plt.legend()
    plt.axis([min(word_ngrams), max(word_ngrams), 0, 1])
    plt.show()

def main():
    #plot_dim()
    plot_epoch()
    #plot_word_ngrams()
    #plot_ws()




if __name__ == '__main__':
    main()