import re
import os

dir = os.path.dirname(__file__)
DOWNLOADED_DICTIONARY_PATH = os.path.join(dir,'Test_Set_3802_Pairs.txt')

downloaded_dictionary = {}
f = open(DOWNLOADED_DICTIONARY_PATH, 'rb')
for word in f:
    word = word.decode('utf8')
    word = word.split()
    downloaded_dictionary[word[1]] = word[3]
f.close()


def clean(tweet):
    """
    Function that cleans the tweet using the functions above and some regular expressions
    to reduce the noise

    Arguments: tweet (the tweet)

    """
    #Separates the contractions and the punctuation


    tweet = re.sub("[!#.,\"]", "", tweet).replace("<user>", "")
    tweet = re.sub("[!#.,\"]", "", tweet).replace("<url>", "")
    tweet = correct_spell(tweet)
    return tweet.strip().lower()

def correct_spell(tweet):
    """
    Function that uses the three dictionaries that we described above and replace noisy words

    Arguments: tweet (the tweet)

    """


    tweet = tweet.split()
    for i in range(len(tweet)):
        if tweet[i] in downloaded_dictionary.keys():
            tweet[i] = downloaded_dictionary[tweet[i]]
    tweet = ' '.join(tweet)
    return tweet

