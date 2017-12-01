#!/usr/bin/env python3
import pickle
import os

dir = os.path.dirname(__file__)
VOCAB_CUT_PATH = os.path.join(dir, '..', '..', 'data', 'preprocessed', 'vocab_cut.txt')
VOCAB_PATH = os.path.join(dir, '..', '..', 'data', 'preprocessed', 'vocab.pkl')
TEST_VOCAB_CUT_PATH = os.path.join(dir, '..', '..', 'data', 'preprocessed', 'test_vocab_cut.txt')
TEST_VOCAB_PATH = os.path.join(dir, '..', '..', 'data', 'preprocessed', 'test_vocab.pkl')

def main():
    vocab = dict()
    with open(VOCAB_CUT_PATH) as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx

    with open(VOCAB_PATH, 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)

    test_vocab = dict()
    with open(TEST_VOCAB_CUT_PATH) as f:
        for idx, line in enumerate(f):
            test_vocab[line.strip()] = idx

    with open(TEST_VOCAB_PATH, 'wb') as f:
        pickle.dump(test_vocab, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
