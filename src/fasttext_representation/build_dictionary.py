import pickle
import numpy as np


def main():
    skipgram = open('model_skipgram.vec', 'r')
    skipgram_dict = {}
    for line in skipgram.readlines():
        tokens = line.split()
        key = tokens[0]
        value = np.array(tokens[1:len(tokens)]).astype(float)
        skipgram_dict[key] = value

    with open('skipgram_dict.pickle', 'wb') as handle:
        pickle.dump(skipgram_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    cbow = open('model_cbow.vec', 'r')
    cbow_dict = {}
    for line in cbow.readlines():
        tokens = line.split()
        key = tokens[0]
        value = np.array(tokens[1:len(tokens)]).astype(float)
        cbow_dict[key] = value

    with open('cbow_dict.pickle', 'wb') as handle:
        pickle.dump(cbow_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
