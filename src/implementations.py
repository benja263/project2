from mean_square import *
from logistic_regression import *
from helpers import *
from proj1_helpers import *

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Applies the gradient descent algorithm with mean square error (MSE).
    
    The step size changes with each iteration and is computed as 1/(n_iter) where n_iter is the current number of iteration.
    
    Args:
        y: The label/output vector (N x 1).
        tx: The data/input matrix (N  x D). Each row represents one instance of data/input.
        initial_w: The initial weight vector (D x 1).
        max_iters: The maximum number of iteration, once reached, the algorithm stops.
        gamma: The initial step size or learning rate.
        
    Returns:
        w: The last iteration of the weight vector (D x 1)
        loss: The last iteration of the loss (MSE here), computed with the last weight
    """
    w = initial_w

    for n_iter in range(max_iters):
        grad, err = compute_gradient_mse(y, tx, w)
        loss = calculate_mse(err)
        w = w - gamma*gradient

    
    return w, loss


def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma)::
    """Applies the stochastic gradient descent algorithm with mean square error (MSE).
    
    The step size changes with each iteration and is computed as 1/(n_iter) where n_iter is the current number of iteration.
    
    Args:
        y: The label/output vector (N x 1).
        tx: The data/input matrix (N  x D). Each row represents one instance of data/input.
        initial_w: The initial weight vector (D x 1).
        max_iters: The maximum number of iteration, once reached, the algorithm stops.
        gamma: The initial step size or learning rate.
        
    Returns:
        w: The last iteration of the weight vector (D x 1)
        loss: The last iteration of the loss (MSE here), computed with the last weight
    
    """
    w = initial_w
    for n in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            stoch_gradient,_ = compute_stoch_gradient_mse(y_batch, tx_batch, w)
            w = w - gamma * stoch_gradient
        
    loss = compute_loss_mse(y,tx,w)

    return w, loss



def least_squares(y, tx):
    """Computes the least squares solution.
    
    Args:
        y: The label/output vector (N x 1).
        tx: The data/input matrix (N  x D). Each row represents one instance of data/input.
        
    Returns:
        The optimal weight vector as the least square solution (N x 1)
        loss: The error (MSE here) computed with the solution weight w
    """

    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    return np.linalg.solve(a, b),compute_loss_mse(y,tx,np.linalg.solve(a, b))


def ridge_regression(y, tx, lambda_):
    """Computes the ridge regression solution.
    
    Args:
        y: The label/output vector (N x 1).
        tx: The data/input matrix (N  x D). Each row represents one instance of data/input.
        lambda_: The ridge regression hyperparameter (scalar)
        
    Returns:
        w: The optimal weight vector as the ridge regression solution (N x 1)
        loss: The error (MSE) computed with the solution weight 
    
    """
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    return np.linalg.solve(a, b), compute_loss_mse(y,tx,np.linalg.solve(a, b))


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Applies the logistic regression algorithm.

    Args:
        y: The label/output vector (N x 1).
        tx: The data/input matrix (N  x D). Each row represents one instance of data/input.
        initial_w: he initial weight vector (D x 1).
        max_iters: The maximum number of iteration.
        gamma: The step size or learning rate.

    Returns:
        w: The trained weights .
        loss: The loss on training set using the trained weights.

    """

    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_loss_logistic(y, tx, w)
        gradient = compute_gradient_logistic(y, tx, w)

        w = w - gamma*gradient


    return w, loss

def newton_logistic_regression(y, tx, initial_w)
    w = initial_w
    for n_iter in range(max_iters):
        loss = calculate_loss(y, tx, w)
        gradient = calculate_gradient(y, tx, w)
        hessian = calculate_hessian(y, tx, w)
        w -= np.linalg.solve(hessian, gradient)
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Applies the regularized/penalized logistic regression algorithm.

    Args:
        y: The label/output vector (N x 1).
        tx: The data/input matrix (N  x D). Each row represents one instance of data/input.
        lambda_: The regularization parameter.
        initial_w: he initial weight vector (D x 1).
        max_iters: The maximum number of iteration
        gamma: The step size or learning rate.

    Returns:
        w: The trained weights .
        loss: The loss on training set using the trained weights.

    """

    w = initial_w

    for n_iter in range(max_iters):
        loss = compute_loss_penalized_logistic(y, tx, w, lambda_)
        gradient = compute_gradient_penalized_logistic(y, tx, w, lambda_)
        w = w - gamma*gradient
    return w, loss

def svm_gds(y, tx, lambda_,initial_w, max_iters, gamma):
    
    w = initial_w
    for n_iter in range(max_iter):
        #n = sample one data point uniformly at random data from x
        n = random.randint(0,num_examples-1)
        grad = calculate_stochastic_gradient(y, X, w, lambda_, n, num_examples)
        cost = calculate_primal_objective(y, X, w, lambda_)
        w -= gamma/(n_iter+1) * grad
    return w,cost
def coordinate_descent(y, tx, lambda_, initial_w , inital_alpha,max_iters):
    w = intial_w
    alpha = initial_alpha
    for n_iter in range(max_iter):
        # n = sample one data point uniformly at random data from x
        n = random.randint(0,num_examples-1)
        w, alpha = calculate_coordinate_update(y, X, lambda_, alpha, w, n)
        cost = calculate_dual_objective(y, X, w, alpha, lambda_)
    return w,cost