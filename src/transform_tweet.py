import numpy as np

def tweetToAvgVec(tweet, vocab, size):
    sum = np.zeros((size))
    for word in tweet:
        if word in vocab:
            sum += vocab[word] # Returns None if the word is not in the vocab
    return sum/len(tweet)

def tweetsToAvgVec(tweets,vocab,size):
    N = len(tweets)
    res = np.zeros((N, size))
    for i in range(N):
        res[i] = tweetToAvgVec(tweets[i], vocab,size)
    return res

def old_tweetToAvgVec(tweet, vocab, embeddings):
    _, K = embeddings.shape
    sum = np.zeros((K))
    words = tweet.split()
    for word in words:
        index = vocab.get(word) # Returns None if the word is not in the vocab
        if index is not None:
            sum += embeddings[index]

    return sum/len(words)

def old_tweetsToAvgVec(tweets, vocab, embeddings):
    N = len(tweets)
    _, K = embeddings.shape
    res = np.zeros((N, K))
    for i in range(N):
        res[i] = old_tweetToAvgVec(tweets[i], vocab, embeddings)

    return res

def create_embedding_matrix(vocab,dict,embedding_dim):
    embedding_matrix = np.zeros((len(dict) + 1,embedding_dim))
    for word, i in dict.items():
        #print(word)
        #print(i)
        if i > len(dict):
            continue
        vec = vocab.get(word)
        if vec is not None:
            # words not found in vocabulary will be all-zeros.
            embedding_matrix[i] = vec
    return embedding_matrix