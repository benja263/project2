import fasttext
import os

d = os.path.dirname(__file__)
TRAIN_PATH = os.path.join(d, 'fasttext_train.txt')
TEST_PATH = os.path.join(d, 'fasttext_validation.txt')

def main():
    classifier = fasttext.supervised(TRAIN_PATH, 'model', dim=100,
                                     epoch=5,
                                     word_ngrams=2,
                                     bucket=2000000,
                                     minn=3,
                                     maxn=6,
                                     ws=10)
    result = classifier.test(TEST_PATH)
    print('P@1:', result.precision)
    print('R@1:', result.recall)
    print('Number of examples:', result.nexamples)


if __name__ == '__main__':
    main()