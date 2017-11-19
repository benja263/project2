from proj1_helpers import *

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def split_data(x, y, ratio, seed=1):
    """Splits the dataset based on the split ratio.

    The function randomly shuffles the dataset before splitting.

    Args:
        x: The data/input matrix (N  x D). Each row represents one instance of data/input.
        y: The label/output vector (N x 1).
        ratio: The ratio of data dedicated to training. Ex: ratio=0.8 will split the dataset into 80% dedicated for training and 20% for testing/validation.
        seed:
    
    """

    np.random.seed(myseed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te

def adjust_labels(y):
    y[np.where(y > 0)] = 1
    y[np.where(y <= 0)] = 0
    return y

def unadjust_labels(y):
    y[np.where(y > 0)] = 1
    y[np.where(y <= 0)] = -1
    return y


def standardize(x):
    """Standardizes the data set by removing means and dividing by standard deviation.
    
    Args:
        x: The dataset to be standardized

    Returns:
        The standardized data set.

    """
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x

def special_standardize(x):
    N,D = x.shape
    bad_index = np.where(x == -999)
    for i in range(D):
        good_index = np.where(x[:,i] != -999)
        if i != 23:
            x[:,i] = x[:,i] - np.mean(x[:,i][good_index])
            x[:,i] = x[:,i] / np.std(x[:,i][good_index])
    x[bad_index] = 0

    return x

def compare_percentage(y1, y2):
    c = 0
    N = len(y1)
    for n in range(N):
        if y1[n] == y2[n]:
            c += 1

    return c/N

def build_poly2(x, degree):
    """
    Returns the matrix formed by applying the polynomial basis to x.
    This implementation only applies the polynomial basis starting from the power of 2 until degree, i.e.degree
    the resulting matrix excludes the power of 0 and 1.


    """
    poly = np.zeros((len(x),degree-1))
    for j in range(2,degree+1):
        poly[:,j-2] = np.power(x,j)  # builds from the power of 2 until degree because we dont want to add multiples to data
        poly[x == - 999, j-2] = -999
    return poly

def add_bias(tx):
    """Adds bias column to a matrix.

    Args:
        tx: The data/input matrix (N  x D). Each row represents one instance of data/input.

    Returns:
        The matrix with the bias column added at the start.

    """

    N,D = tx.shape
    tx = np.hstack((np.ones((N, 1)), tx))
    return tx

def construct_poly_basis(tx, degree):
    """ Applies a polynomial basis to the dataset.

    The idea is to first construct the extra columns of the polynomial basis and then stack them with the dataset iteratively.

    For example:
    If degree is 2, then we build the square of each column and then add stack them with the dataset iteratively.

    """


    N,D = tx.shape
    res = np.copy(tx)
    for feature in range(D):
        if feature != 23:
            col = tx[:, feature]
            poly = build_poly2(col, degree)
            res = np.hstack((res,poly))

    return res

def substitute(x, old, new):
    """Substiute any occurence of a value by another one.

    Args:
        x: The matrix or vector where we want to apply the substitution.
        old: The value to be replaced
        new: The new value

    Returns:
        The application of the substitution on x.

    """
    copy = np.copy(x)
    copy[copy == old] = new
    return copy

def add_bias_to_best_features_index(best_features):
    """Adds the bias column index to the sorted index list of best features.best

    The bias column is added as the best feature (always the first column)

    Args:
        best_features: Vector holding the index of the features ordered from best to by Fisher score

    Returns:
        The index of the features ordered from best to worst by Fisher score.
        The bias column is added as the first element.
    """

    best_features += 1 # 
    best_features = np.hstack(([0], best_features))
    return best_features

def standardize_ignore_value(x, value):
    """
    N,D = x.shape
    for i in range(D):
        x[i] = x[i]-np.mean()
    """
    return 0

def remove_col(x, n):
    return np.delete(x, n, 1)

def predict_labels_logistic(weights, data):
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = 0
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred
