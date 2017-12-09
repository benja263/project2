import os

dir = os.path.dirname(__file__)
POS_PATH = os.path.join(dir, '..', '..', 'data', 'raw', 'train_pos.txt')
NEG_PATH = os.path.join(dir, '..', '..', 'data', 'raw', 'train_neg.txt')

"""
 This scripts create the train and test/validation file for fasttext.
 Each training sample needs to be prefixed with __label__<label>.
"""

def main():
    fasttext_train = open('fasttext_train.txt', 'w') # Training file
    fasttext_validation = open('fasttext_validation.txt', 'w') # Validation/Test file

    with open(POS_PATH) as f:
        lines = f.readlines()
        cutoff = round(len(lines)*0.7)

        i = 0
        for line in lines:
            if i < cutoff:
                fasttext_train.write('__label__1 ' + line + '\n')
            else:
                fasttext_validation.write('__label__1 ' + line + '\n')
            i += 1

    with open(NEG_PATH) as f:
        lines = f.readlines()

        cutoff = round(len(lines) * 0.7)

        j = 0
        for line in lines:
            if j < cutoff:
                fasttext_train.write('__label__-1 ' + line + '\n')
            else:
                fasttext_validation.write('__label__-1 ' + line + '\n')
            j += 1

if __name__ == '__main__':
    main()