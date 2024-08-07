import numpy as np
import time
import pandas as pd
from scipy.optimize import minimize, check_grad
from scipy.optimize import lsq_linear


def layer_sizes(X, Y, number_neurons=4):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)

    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """

    n_x = X.shape[0]  # size of input layer
    n_h = number_neurons
    n_y = Y.shape[0]  # size of output layer
    return (n_x, n_h, n_y)

def tanh(x, spread=1):
    y = (np.exp(2 * spread * x) - 1) / (np.exp(2 * spread * x) + 1)
    return y

def compute_Z1(X, W, spread):  # this function computes Z1,once we have X, W SPREAD,which are all parametres
    A1 = np.dot(W, X)
    Z1 = tanh(A1, spread)
    return np.array(Z1)

def cost_function(weights, argsfun):
    # weights is a column vector containing just V, so the dimension is n_h
    # argsfun is a list containing [X,Y,W,spread,L2]
    X, Y = argsfun[0], argsfun[1]
    W, spread = argsfun[2], argsfun[3]
    L2 = argsfun[4]
    Z1 = compute_Z1(X, W, spread)
    V = weights.reshape(1, -1)
    P = len(Y[0])
    Z2 = np.dot(V, Z1)
    cost = (1 / (2 * P)) * np.square(np.linalg.norm(Z2 - Y)) + (L2 / 2) * np.square(np.linalg.norm(weights))
    return cost

def minimize(X_Train, Y_Train, number_neurons, spread, L2):
    n_x, n_h, n_y = layer_sizes(X_Train, Y_Train, number_neurons)
    matrix_b = np.append(Y_Train.T, np.zeros(n_h).reshape(-1, 1), axis=0).reshape(-1)
    P = len(Y_Train[0])
    lambda_tilde = np.sqrt(L2 * P)

    W = np.random.normal(0.17642471764481418, 0.5540804796003468, n_h * n_x).reshape(n_h, n_x)
    Z1 = compute_Z1(X_Train, W, spread).T
    matrix_A = np.append(Z1, lambda_tilde * np.identity(n_h), axis=0)
    t0 = time.time()
    res = lsq_linear(matrix_A, matrix_b, lsmr_tol='auto', verbose=0)
    opt_time = time.time() - t0

    Training_regularized_error = res["cost"] / 186
    param = [X_Train, Y_Train, W, spread, 0]  # in order to calculate error without regolarization
    V = res.x
    Training_non_regularized_error = cost_function(V, param)
    d = {"W": W, "V": V, "Training_regularized_error": Training_regularized_error,
         "Training_non_regularized_error": Training_non_regularized_error, 'time': opt_time, 'message': res.message,
         'iter': res.nit}

    return d

def predict(V,args, err=True):
    #V is the only vector of variables
    #args is a list containing[Xtest,Ytest,W,spread]
    X, Y = args[0], args[1]
    W, spread = args[2], args[3]
    Z1 = compute_Z1(X, W, spread)
    y_prediction = np.dot(V.T,Z1)
    if err:
        parameters = [X, Y, W, spread, 0]
        test_error = cost_function(V, parameters)
    else:
        test_error = None
    return y_prediction, test_error





