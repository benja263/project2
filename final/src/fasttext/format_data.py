import os

d = os.path.dirname(__file__)

POS_PATH = os.path.join(d, '..', '..', 'data', 'raw', 'train_pos_full.txt')
NEG_PATH = os.path.join(d, '..', '..', 'data', 'raw', 'train_neg_full.txt')
FILTERED_POS_PATH = os.path.join(d, 'text', 'filtered_train_pos_full.txt')
FILTERED_NEG_PATH = os.path.join(d, 'text', 'filtered_train_neg_full.txt')
SW_POS_PATH = os.path.join(d, 'text', 'sw_train_pos_full.txt')
SW_NEG_PATH = os.path.join(d, 'text', 'sw_train_neg_full.txt')

TRAIN_PATH = os.path.join(d, 'text', 'fasttext_train.txt')
VALIDATION_PATH = os.path.join(d, 'text', 'fasttext_validation.txt')
FINAL_PATH = os.path.join(d, 'text', 'fasttext_train_final.txt')

FILTERED_TRAIN_PATH = os.path.join(d, 'text', 'filtered_fasttext_train.txt')
FILTERED_VALIDATION_PATH = os.path.join(d, 'text', 'filtered_fasttext_validation.txt')
FILTERED_FINAL_PATH = os.path.join(d, 'text', 'filtered_fasttext_train_final.txt')

SW_TRAIN_PATH = os.path.join(d, 'text', 'sw_fasttext_train.txt')
SW_VALIDATION_PATH = os.path.join(d, 'text', 'sw_fasttext_validation.txt')
SW_FINAL_PATH = os.path.join(d, 'text', 'sw_fasttext_train_final.txt')

"""
 This scripts create the train and test/validation file for fasttext.
 Each training sample needs to be prefixed with __label__<label>.
"""

def format(pos_path, neg_path, train_path, validation_path, final_path, train_ratio):
    """
    This function reads the positive dataset and negative dataset and formats each tweet to be usable by fastText. Then
    it writes the formatted tweet to three files:
    - Two files used cross validation: train and validation.
    - One file used for the final prediction: final.

    Each formatted tweet is to either the training set file or the validation set file according to the
    training_ratio parameter. Additionally the function also writes each formatted tweet to the final training file, which will be used
    to do prediction.

    Fasttext requires each data point to be labeled with the prefix '__label__<value>' to be usable by the classifier.
    In our case, we format a positive tweet to have the label '__label__1' and a negative tweet to have the label '__label__-1'.


    :param pos_path: The path to the positive training set file
    :param neg_path: The path to the negative training set file
    :param train_path: The path where the formatted training set will be written
    :param validation_path: The path where the formatted validation set will be written
    :param final_path: The path where the formatted training set used for prediction will be written
    :param train_ratio: The ratio training set ratio, Ex: 0.7 means 70% of the raw dataset will be used for training and the other 30% for validation.
    :return: This function does not return anything
    """

    train = open(train_path, 'w')
    validation = open(validation_path, 'w')
    final = open(final_path, 'w')

    with open(pos_path) as f:
        lines = f.readlines()
        cutoff = round(len(lines) * train_ratio)

        i = 0
        for line in lines:
            formatted_line = '__label__1 ' + line + '\n'
            if i < cutoff:
                train.write(formatted_line)
            else:
                validation.write(formatted_line)
            i += 1

            final.write(formatted_line)

    with open(neg_path) as f:
        lines = f.readlines()

        cutoff = round(len(lines) * 0.7)

        j = 0
        for line in lines:
            formatted_line = '__label__-1 ' + line + '\n'
            if j < cutoff:
                train.write(formatted_line)
            else:
                validation.write(formatted_line)
            j += 1

            final.write(formatted_line)

def main():
    format(POS_PATH, NEG_PATH, TRAIN_PATH, VALIDATION_PATH, FINAL_PATH, 0.7)
    format(FILTERED_POS_PATH, FILTERED_NEG_PATH, FILTERED_TRAIN_PATH, FILTERED_VALIDATION_PATH, FILTERED_FINAL_PATH, 0.7)
    format(SW_POS_PATH, SW_NEG_PATH, SW_TRAIN_PATH, SW_VALIDATION_PATH, SW_FINAL_PATH, 0.7)

if __name__ == '__main__':
    main()