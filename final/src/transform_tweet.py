import numpy as np

def tweetToAvgVec(tweet, vocab, embeddings):
    """
    Returns the average word vector of a tweet.
    The average word vector is computed as the average of the vector representation of the tweet's word.
    If a word does not appear in the vocab, then we add a zero vector.

    :param tweet: The tweet to be converted
    :param vocab: The vocab as a dictionary
    :param embeddings: The embeddings matrix containing the vector representation of words
    :return: The average word vector of a tweet
    """
    _, K = embeddings.shape
    sum = np.zeros((K))
    words = tweet.split()
    for word in words:
        index = vocab.get(word) # Returns None if the word is not in the vocab
        if index is not None:
            sum += embeddings[index]

    return sum/len(words)

def tweetsToAvgVec(tweets, vocab, embeddings):
    """
    Computes the average word vector of multiple tweets.
    Iterates through the tweets and calls tweetToAvgVec for each of them.

    :param tweets: The tweets to be converted
    :param vocab: The vocabulary as a dictionary
    :param embeddings: The embeddings matrix containing the vector representation of words
    :return: The average word vectors of the tweets
    """

    N = len(tweets)
    _, K = embeddings.shape
    res = np.zeros((N, K))
    for i in range(N):
        res[i] = tweetToAvgVec(tweets[i], vocab, embeddings)

    return res

def create_embedding_matrix(vocab, dict, embedding_dim):
    """
    Creates a new embedding matrix from a dictionary.
    For each word in the dictionary, we check that it exists in our vocabulary.
    If it does, then we write the vector of this word at the corresponding index in the dictionary.

    :param vocab: A matrix containing the vector representation of words
    :param dict: A dictionary mapping a word to an index
    :param embedding_dim: The feature dimension of the embedding matrix
    :return: A new embedding matrix
    """

    embedding_matrix = np.zeros((len(dict) + 1, embedding_dim))
    for word, i in dict.items():
        if i > len(dict):
            continue
        vec = vocab.get(word)
        if vec is not None:
            # words not found in vocabulary will be all-zeros.
            embedding_matrix[i] = vec
    return embedding_matrix

def create_word2Vec_embedding_matrix(vocab, dict, embedding_dim):
    """
    Same as create_embedding_matrix but for Word2Vec.
    The dict of Word2Vec uses a special structure.

    :param vocab:  matrix containing the vector representation of words
    :param dict: Word2Vec dict
    :param embedding_dim: The feature dimension of the embedding matrix
    :return: A new embedding matrix based on Word2Vec
    """
    embedding_matrix = np.zeros((len(dict) + 1, embedding_dim))
    for word, i in dict.items():
        if i > len(dict):
            continue
        try:
            vec = vocab[word]
        except KeyError:
            vec = np.zeros((embedding_dim))
        # words not found in vocabulary will be all-zeros.
        embedding_matrix[i] = vec
    return embedding_matrix