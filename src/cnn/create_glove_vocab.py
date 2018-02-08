import os
import numpy as np
import pickle

d = os.path.dirname(__file__)
VOCAB_PATH = os.path.join(d, 'glove.6B.300d.txt')


def main():
    print('Loading Vocabulary')
    with open(VOCAB_PATH) as f:
        txt_lines = f.readlines()
    vocab = {}
    for line in txt_lines:
        tokens = line.split()
        key = tokens[0]
        value = np.asarray(tokens[1:len(tokens)], dtype='float32')
        vocab[key] = value
    with open('glove_dict.pickle', 'wb') as handle:
        pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

if __name__ == '__main__':
    main()