import numpy as np

def sigmoid(t):
    """Applies the sigmoid function.
    
    Args:
        t: The input, can be a scalar, vector or matrix
        
    Returns:
        The application of the sigmoid function to t
        
    """

    return 1.0 / (1 + np.exp(-t))


def compute_loss_logistic(y, tx, w):
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)


def compute_gradient_logistic(y, tx, w):
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad


def compute_loss_penalized_logistic(y, tx, w, lambda_):
    return compute_loss_logistic(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))


def compute_gradient_penalized_logistic(y, tx, w, lambda_):
    return compute_gradient_logistic(y, tx, w)  + 2 * lambda_ * w

def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    pred = sigmoid(tx.dot(w))
    pred = np.diag(pred.T[0])
    r = np.multiply(pred, (1-pred))
    return tx.T.dot(r).dot(tx)

