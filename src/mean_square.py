import numpy as np
from proj1_helpers import predict_labels

def compute_loss_mse(y, tx, w):
    """Computes the mean square error (MSE) loss function.

    Args:
        y: The label/output vector (N x 1).
        tx: The data/input matrix (N  x D). Each row represents one instance of data/input.
        w: The weight vector (D x 1).

    Returns:
        The mean square error (MSE)
    """
    e = y - tx.dot(w)
    return calculate_mse(e)

def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)

def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))

def compute_gradient_mse(y, tx, w):
    """Computes the gradient of the mean square error (MSE) loss function.
    
    Args:
        y: The label/output vector (N x 1).
        tx: The data/input matrix (N  x D). Each row represents one instance of data/input.
        w: The weight vector (D x 1).
        
    Returns:
        The gradient of the mean square error loss function (MSE) (D x 1)

    """
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


def compute_stoch_gradient_mse(y, tx, w):
    """Computes the stochastic gradient of the mean square error (MSE) loss function.
    
    Args:
        y: A randomly chosen (by stochastic process) label/output (1 x 1).
        tx: The corresponding input data vector (1 x D).
        w: The weight vector (D x 1).
        
    Returns:
        gradient: The stochastic gradient of the mean square error loss function (MSE) (D x 1)

    """

    return compute_gradient_mse(y,tx,w)