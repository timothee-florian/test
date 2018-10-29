import numpy as np
from plots import cross_validation_visualization
from proj1_helpers import *
#functions

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

def sigmoid(x):
    return 1/(1+np.exp(-x))

def compute_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)

def compute_gradient_mse(y, x, w):
    """Compute the mse gradient."""
    N= x.shape[0]
    e = y - np.matmul(x, w)
    return  -1 * np.dot(x.T, e) /N

def compute_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))

def compute_logi_loss(y, x, w):
    """Compute the logistic loss of"""
    y_t = sigmoid(x.dot(w))
    N= len(y)
    loss1 = y.T.dot(np.log(y_t+10**-18))
    loss2 = (1- y).T.dot(np.log(1-y_t+10**-18))
    return -1/N*(loss1+loss2)

def compute_error(y, x, w, error):
    if error == 'mse':
        e = y - x.dot(w)
        return compute_mse(e)
    elif error == 'mae':
        e = y - x.dot(w)
        return compute_mae(e)

def grad_logistic(y, x, w, lambda_1, lambda_2):
    """Compute de gradient for logistic regression with regularization L1 and L2"""
#     N= x.shape[0]
    z = sigmoid(x.dot(w))
    return x.T.dot(z-y) + np.sign(w) * lambda_1 + w * lambda_2


# def least_squares(y, tx):
#     """calculate the least squares solution"""
#     a = tx.T.dot(tx)
#     b = tx.T.dot(y)
#     w = np.linalg.solve(a, b)   
#     return w 
    
def ridge_regression_solu(y, x, lambda_):
    """implement ridge regression."""
    aI = 2 * x.shape[0] * lambda_ * np.identity(x.shape[1])
    a = x.T.dot(x) + aI
    b = x.T.dot(y)
    return np.linalg.solve(a, b)

def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
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
    if len(x_tr.shape)<2:
        return x_tr.reshape(-1,1), x_te.reshape(-1,1), y_tr.reshape(-1,1), y_te.reshape(-1,1)
    return x_tr, x_te, y_tr.reshape(-1,1), y_te.reshape(-1,1)


def gradient_descent(y, x, initial_w, max_iters, gamma, lambda_1, lambda_2,\
                                             compute_gradient, compute_loss):
    """Gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        dw = compute_gradient(y, x, w, lambda_1, lambda_2)
        w = w - gamma * dw
    loss = compute_loss(y, x, w)
    return w, loss


def stochastic_gradient_descent(y, x, initial_w, batch_size, max_iters, gamma,\
                             lambda_1, lambda_2, compute_gradient, compute_loss, return_last=True):
    """Stochastic gradient descent."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, x, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            dw = compute_gradient(y_batch, tx_batch, w, lambda_1, lambda_2)
            # update w through the stochastic gradient update
            w = w - gamma * dw
            # calculate loss
            loss = compute_loss(y, x, w)
            # store w and loss
            ws.append(w)
            losses.append(loss)

        #uncomment to see the loss progression
        # print("SGD({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))
    if return_last:
        return ws[-1] , losses[-1]
    return  ws, losses

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, lambda_, degree, regression):
    """return the loss of ridge regression or logistic regression"""
    # ***************************************************
    # select training and testing sets
    y_test = y[k_indices[k]].reshape(-1,1)
    x_test =x[k_indices[k]]
    k_indices=np.delete(k_indices, k)
    y_train = y[k_indices].reshape(-1,1)
    x_train =x[k_indices]
    # ***************************************************
    # standardize the set
    x_train, mean_x, std_x= standardize(x_train)
    x_test = (x_test-mean_x)/std_x
    x_train = build_poly(x_train, degree)
    x_test = build_poly(x_test, degree)
    # ***************************************************
    # train the chosen model
    print(y_train.shape)
    print(regression)
    if regression == 'ridge_regression':
        w = ridge_regression_solu(y_train, x_train, lambda_)
        # compute and return the losses
        loss_tr = compute_error(y_train, x_train, w, 'mse')
        loss_te = compute_error(y_test, x_test, w, 'mse')
    else:
        print(y_train.shape)
        y_train[y_train<0]=0
        y_test[y_test<0]=0
        w_initial = np.zeros(x_train.shape[1],1)
        w, loss = gradient_descent(y_train, x_train, w_initial, max_iters= 200, gamma=0.001, lambda_1=0,
                                   lambda_2=lambda_, compute_gradient=grad_logistic, compute_loss = compute_logi_loss)
        # compute and return the losses
        loss_tr = compute_logi_loss(y_train, x_train, w)
        loss_te = compute_logi_loss(y_test, x_test, w)

    return loss_tr, loss_te


def cross_validation_accross_lambdas(x, y, regression, degree = 3, k_fold = 4, lambdas = np.logspace(-5, 3, 8)):
    seed = 2
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    mean_losses_tr = []
    mean_losses_te = []
    # ***************************************************
    for lambda_ in lambdas:
        loss_train = 0
        loss_test = 0
        for k in range(k_fold):
            loss_tr, loss_te = cross_validation(y, x, k_indices, k, lambda_, degree, regression)
            loss_train += loss_tr
            loss_test += loss_te
        #append the losses
        mean_losses_tr += [loss_train/k_fold]
        mean_losses_te += [loss_test/k_fold]
    # ***************************************************
    cross_validation_visualization(xs=lambdas, mse_tr= mean_losses_tr, mse_te=mean_losses_te, x_name='lambda')
    return mean_losses_tr, mean_losses_te

def cross_validation_accross_degrees(x, y, regression, degrees = range(1,10), k_fold = 4, lambda_ = 0):
    seed = 2
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    mean_losses_tr = []
    mean_losses_te = []
    # ***************************************************

    for degree in degrees:
        loss_train = 0
        loss_test = 0
        for k in range(k_fold):
            loss_tr, loss_te = cross_validation(y, x, k_indices, k, lambda_, degree, regression)   
            loss_train += loss_tr
            loss_test += loss_te
        #append the losses with the mean error
        mean_losses_tr += [loss_train/k_fold]
        mean_losses_te += [loss_test/k_fold]
    # ***************************************************  
    cross_validation_visualization(xs=degrees, mse_tr= mean_losses_tr, mse_te=mean_losses_te, x_name='degree', xscale='lin')
    return mean_losses_tr, mean_losses_te