#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle
import random
import os

dir = os.path.dirname(__file__)
COOC_PATH = os.path.join(dir, '..', '..', 'data', 'preprocessed', 'cooc.pkl')
SAVE_PATH = os.path.join(dir, '..', '..', 'data', 'embeddings')
TEST_COOC_PATH = os.path.join(dir, '..', '..', 'data', 'preprocessed', 'test_cooc.pkl')
TEST_SAVE_PATH = os.path.join(dir, '..', '..', 'data', 'test_embeddings')

def main():

    print("loading cooccurrence matrix")
    with open(COOC_PATH, 'rb') as f:
        cooc = pickle.load(f)
    print("{} nonzero entries".format(cooc.nnz))

    nmax = 100
    print("using nmax =", nmax, ", cooc.max() =", cooc.max())

    print("initializing embeddings")
    embedding_dim = 20
    xs = np.random.normal(size=(cooc.shape[0], embedding_dim))
    ys = np.random.normal(size=(cooc.shape[1], embedding_dim))

    eta = 0.001
    alpha = 3 / 4

    epochs = 10

    for epoch in range(epochs):
        print("epoch {}".format(epoch))
        for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
            logn = np.log(n)
            fn = min(1.0, (n / nmax) ** alpha)
            x, y = xs[ix, :], ys[jy, :]
            scale = 2 * eta * fn * (logn - np.dot(x, y))
            xs[ix, :] += scale * y
            ys[jy, :] += scale * x
    np.save(SAVE_PATH, xs)

    print('TEST')
    print("loading cooccurrence matrix")
    with open(TEST_COOC_PATH, 'rb') as f:
        test_cooc = pickle.load(f)
    print("{} nonzero entries".format(test_cooc.nnz))

    nmax = 100
    print("using nmax =", nmax, ", cooc.max() =", test_cooc.max())

    print("initializing embeddings")
    embedding_dim = 20
    xs = np.random.normal(size=(test_cooc.shape[0], embedding_dim))
    ys = np.random.normal(size=(test_cooc.shape[1], embedding_dim))

    eta = 0.001
    alpha = 3 / 4

    epochs = 10

    for epoch in range(epochs):
        print("epoch {}".format(epoch))
        for ix, jy, n in zip(test_cooc.row, test_cooc.col, test_cooc.data):
            logn = np.log(n)
            fn = min(1.0, (n / nmax) ** alpha)
            x, y = xs[ix, :], ys[jy, :]
            scale = 2 * eta * fn * (logn - np.dot(x, y))
            xs[ix, :] += scale * y
            ys[jy, :] += scale * x
    np.save(TEST_SAVE_PATH, xs)


if __name__ == '__main__':
    main()
