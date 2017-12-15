#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle
import os

# Use relative path to this file instead of relative path to the caller
dir = os.path.dirname(__file__)
POS_TRAIN_PATH = os.path.join(dir, '..', '..', 'data', 'raw', 'train_pos.txt')
NEG_TRAIN_PATH = os.path.join(dir, '..', '..', 'data', 'raw', 'train_neg.txt')
VOCAB_PATH = os.path.join(dir, '..', '..', 'data', 'preprocessed', 'vocab.pkl')
COOC_PATH = os.path.join(dir, '..', '..', 'data', 'preprocessed', 'cooc.pkl')
TEST_PATH = os.path.join(dir, '..', '..', 'data', 'preprocessed', 'parsed_test_data.txt')
TEST_VOCAB_PATH = os.path.join(dir, '..', '..', 'data', 'preprocessed', 'test_vocab.pkl')
TEST_COOC_PATH = os.path.join(dir, '..', '..', 'data', 'preprocessed', 'test_cooc.pkl')


def main():
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)

    data, row, col = [], [], []
    counter = 1
    for fn in [POS_TRAIN_PATH, NEG_TRAIN_PATH]:
        with open(fn) as f:
            for line in f:
                tokens = [vocab.get(t, -1) for t in line.strip().split()]
                tokens = [t for t in tokens if t >= 0]
                for t in tokens:
                    for t2 in tokens:
                        data.append(1)
                        row.append(t)
                        col.append(t2)

                if counter % 10000 == 0:
                    print(counter)
                counter += 1
    cooc = coo_matrix((data, (row, col)))
    print("summing duplicates (this can take a while)")
    cooc.sum_duplicates()
    with open(COOC_PATH, 'wb') as f:
        pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
