import numpy as np

def tweetToAvgVec(tweet, vocab, embeddings):
    _, K = embeddings.shape
    sum = np.zeros((K))
    words = tweet.split()

    for word in words:
        index = vocab.get(word) # Returns None if the word is not in the vocab
        if index is not None:
                sum += embeddings[index]

    return sum/len(words)

def tweetsToAvgVec(tweets, vocab, embeddings):
    N = len(tweets)
    _, K = embeddings.shape
    res = np.zeros((N, K))

    for i in range(N):
        res[i] = tweetToAvgVec(tweets[i], vocab, embeddings)

    return res
