import os
from nltk.corpus import stopwords
import re

d = os.path.dirname(__file__)
POS_PATH = os.path.join(d, '..', '..', 'data', 'raw', 'train_pos_full.txt')
NEG_PATH = os.path.join(d, '..', '..', 'data', 'raw', 'train_neg_full.txt')
TEST_PATH = os.path.join(d, '..', '..', 'data', 'preprocessed', 'parsed_test_data.txt')

FILTERED_POS_FILE_PATH = os.path.join(d, 'text', 'filtered_train_pos_full.txt')
FILTERED_NEG_FILE_PATH = os.path.join(d, 'text', 'filtered_train_neg_full.txt')
FILTERED_TEST_FILE_PATH = os.path.join(d, 'text', 'filtered_test_full.txt')

SW_POS_FILE_PATH = os.path.join(d, 'text', 'sw_train_pos_full.txt')
SW_NEG_FILE_PATH = os.path.join(d, 'text', 'sw_train_neg_full.txt')
SW_TEST_FILE_PATH = os.path.join(d, 'text', 'sw_test_full.txt')


def clean_tweet(tweet):
    """
    Cleans a tweet by removing any non-alphanumerical characters.

    :param tweet: The tweet to be cleaned
    :return: The cleaned tweet
    """

    tweet = re.sub("[^A-Za-z0-9 \n]+", "", tweet)
    return tweet


def clean_tweet_stop_words(tweet, sw):
    """
    Cleans a tweet by removing any non-alphanumerical characters and stop words.

    :param tweet: The tweet to be cleaned
    :param sw: A set object containing the stop words
    :return: The cleaned tweet
    """

    cleaned = clean_tweet(tweet)
    words = cleaned.split(" ")
    return " ".join([word for word in words if word not in sw])

def main():
    filtered_pos_file = open(FILTERED_POS_FILE_PATH, 'w')
    filtered_neg_file = open(FILTERED_NEG_FILE_PATH, 'w')
    filtered_test_file = open(FILTERED_TEST_FILE_PATH, 'w')

    sw_pos_file = open(SW_POS_FILE_PATH, 'w')
    sw_neg_file = open(SW_NEG_FILE_PATH, 'w')
    sw_test_file = open(SW_TEST_FILE_PATH, 'w')

    sw = set(stopwords.words('english'))

    with open(POS_PATH) as f:
        tweets = f.readlines()
        for tweet in tweets:
            filtered_pos_file.write(clean_tweet(tweet))
            sw_pos_file.write(clean_tweet_stop_words(tweet, sw))

    with open(NEG_PATH) as f:
        tweets = f.readlines()

        for tweet in tweets:
            filtered_neg_file.write(clean_tweet(tweet))
            sw_neg_file.write(clean_tweet_stop_words(tweet, sw))

    with open(TEST_PATH) as f:
        tweets = f.readlines()

        for tweet in tweets:
            filtered_test_file.write(clean_tweet(tweet))
            sw_test_file.write(clean_tweet_stop_words(tweet, sw))

if __name__ == '__main__':
    main()