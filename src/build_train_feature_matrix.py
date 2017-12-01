import numpy as np
import pickle
import os

# Use relative path to this file instead of relative path to the caller
dir = os.path.dirname(__file__)
POS_TRAIN_PATH = os.path.join(dir, '..', '..', 'data', 'raw', 'train_pos.txt')
NEG_TRAIN_PATH = os.path.join(dir, '..', '..', 'data', 'raw', 'train_neg.txt')
VOCAB_PATH = os.path.join(dir, '..', '..', 'data', 'preprocessed', 'vocab.pkl')
COOC_PATH = os.path.join(dir, '..', '..', 'data', 'preprocessed', 'cooc.pkl')

def main():
    print('Loading Vocabulary')
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
    print('Loading Vocabulary')
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)




if __name__ == '__main__':
    main()