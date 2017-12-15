#!/usr/bin/env python3
import pickle
import os

dir = os.path.dirname(__file__)
VOCAB_CUT_PATH = os.path.join(dir, '..', '..', 'data', 'preprocessed', 'vocab_cut.txt')
VOCAB_PATH = os.path.join(dir, '..', '..', 'data', 'preprocessed', 'vocab.pkl')


def main():
    vocab = dict()
    with open(VOCAB_CUT_PATH) as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx

    with open(VOCAB_PATH, 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
