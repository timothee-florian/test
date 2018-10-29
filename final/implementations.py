
import numpy as np
from functions import *
from proj1_helpers import *


def least_squares_GD(y, x, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        dw = compute_gradient_mse(y, x, w)
        w = w - gamma * dw
    loss = compute_error(y, x, w, 'mse')
    return w, loss

def least_squares_SGD(y, x, initial_w, max_iters, gamma):
    """Stochastic gradient descent."""
    # Define parameters to store w and loss
    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, x, batch_size=1, num_batches=1):
            # compute a stochastic gradient and loss
            dw = compute_gradient_mse(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * dw
            # calculate loss
    loss = compute_error(y, x, w, 'mse')
    return  w, loss

def least_squares(y, x):
    """calculate the least squares solution"""
    a = x.T.dot(tx)
    b = x.T.dot(y)
    w = np.linalg.solve(a, b)   
    loss = compute_error(y, x, w, 'mse')
    return w , loss

def ridge_regression(y, x, lambda_):
    """implement ridge regression."""
    aI = 2 * x.shape[0] * lambda_ * np.identity(x.shape[1])
    a = x.T.dot(x) + aI
    b = x.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_error(y, x, w, 'mse')
    return w , loss

def logistic_regression(y, x, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        dw = grad_logistic(y, x, w, lambda_1=0, lambda_2=0)
        w = w - gamma * dw
    loss = compute_logi_loss(y, x, w)
    return w, loss

def reg_logistic_regression(y, x, lambda_, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        dw = grad_logistic(y, x, w, lambda_1=0, lambda_2=lambda_)
        w = w - gamma * dw
    loss = compute_logi_loss(y, x, w)
    return w, loss
