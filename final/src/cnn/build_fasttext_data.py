import os

dir = os.path.dirname(__file__)
POS_PATH = os.path.join(dir, '..', '..', 'data', 'raw', 'train_pos_full.txt')
NEG_PATH = os.path.join(dir, '..', '..', 'data', 'raw', 'train_neg_full.txt')


def main():
    fasttext_train = open('fasttext_train_full.txt', 'w') # Training file

    with open(POS_PATH) as f:
        lines = f.readlines()

        for line in lines:
                fasttext_train.write('__label__1 ' + line + '\n')

    with open(NEG_PATH) as f:
        lines = f.readlines()

        for line in lines:
                fasttext_train.write('__label__-1 ' + line + '\n')

if __name__ == '__main__':
    main()