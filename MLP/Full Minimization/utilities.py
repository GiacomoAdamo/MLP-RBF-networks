import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize, check_grad


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

def derivata_tanh(x, spread=1):
    # (4 E^(2 s x) s)/(1 + E^(2 s x))^2
    return (4 * spread * np.exp(2 * spread * x)) / ((np.exp(2 * spread * x) + 1) ** 2)

def params_col(parameters):
    param1 = parameters['W'].reshape((-1, 1))
    param2 = parameters['V'].reshape((-1, 1))
    param = np.concatenate((param1, param2))
    return param

def reverse_params_col(param, n_x, n_h, n_y):
    W = param[0:n_x * n_h].reshape((n_h, n_x))
    V = param[n_x * n_h:].reshape((n_y, n_h))
    return W, V

def forward_propagation(X, parameters, spread=1):
    """
    Argument:
    X -- input data of size (m, n_x)
    parameters -- python dictionary containing your parameters (output of initialization function)

    Returns:
    A2 -- The linear output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    W = parameters['W']
    V = parameters['V']

    # Implement Forward Propagation to calculate A2
    A1 = np.dot(W, X)  
    Z1 = tanh(A1, spread)  
    A2 = np.dot(V, Z1)
    Z2 = A2

    cache = {"A1": A1, "Z1": Z1, "A2": A2, "Z2": Z2}

    return Z2, cache

def cost_fun(weights, argsfun):
    """
    weights -- input vector of dimension (n_weight_tot, )
    argsfun -- list of parameters [X, Y, spread, L2, n_x, n_h, n_y]
    ----------------------------------
    output -- scalar value of regularized cost
    """
    X, Y = argsfun[0], argsfun[1]
    spread, L2 = argsfun[2], argsfun[3]
    n_x, n_h, n_y = argsfun[4], argsfun[5], argsfun[6]

    W, V = reverse_params_col(weights.reshape((-1, 1)), n_x, n_h, n_y)  # NB must transform weights from (n_weight_tot, )
    parameters = {"W": W, "V": V}  # to (n_weight_tot, 1)

    # recall foward propagation
    Z2, cache = forward_propagation(X, parameters, spread)

    # compute the cost
    m = Y.shape[1]  # number of example

    # Compute the regularized cost
    cost = (1 / (2 * m)) * np.square(np.linalg.norm(Z2 - Y)) + (L2 / 2) * np.linalg.norm(params_col(parameters)) ** 2
    # (L2 / 2 * m) * np.sum(np.square(parameters))

    cost = np.squeeze(cost)

    return cost

def grad_bp(weights, argsfun):
    """
    weights -- input vector of dimension (n_weight_tot, )
    argsfun -- list of parameters [X, Y, spread, L2, n_x, n_h, n_y]
    ----------------------------------
    grads -- vector of gradients with the following shape (n_weight_tot, )
    """
    X, Y = argsfun[0], argsfun[1]
    spread, L2 = argsfun[2], argsfun[3]
    n_x, n_h, n_y = argsfun[4], argsfun[5], argsfun[6]

    W, V = reverse_params_col(weights.reshape((-1, 1)), n_x, n_h, n_y)
    parameters = {"W": W, "V": V}

    # recall foward propagation
    Z2, cache = forward_propagation(X, parameters, spread)

    # backpropagation
    A1 = cache["A1"]
    Z1 = cache["Z1"]
    m = len(Y[0])
    lamda = L2

    E = Z2 - Y
    dV = (1 / m) * np.dot(E, Z1.T) + lamda * V
    delta = np.multiply(derivata_tanh(A1, spread), (np.dot(V.T, E)))
    dW = (1 / m) * np.dot(delta, X.T) + lamda * W
    gradients = np.concatenate((dW.reshape((-1, 1)), dV.reshape((-1, 1))))
    grads = gradients.reshape((-1))

    return grads

def shallow_NN_train(X_Train, Y_Train, X_val, Y_val, number_neurons, spread, L2, max_iter, validate=True):
    n_x, n_h, n_y = layer_sizes(X_Train, Y_Train, number_neurons)

    weights_0 = np.random.rand(n_x * n_h + n_h * n_y)
    args = [X_Train, Y_Train, spread, 0, n_x, n_h, n_y]
    Training_error_0 = cost_fun(weights_0, args)
    init_norm = np.linalg.norm(grad_bp(weights_0, args))

    args = [X_Train, Y_Train, spread, L2, n_x, n_h, n_y]

    options = {'maxiter': max_iter}
    t0 = time.time()
    res = minimize(cost_fun, weights_0, args=args, method='BFGS', jac=grad_bp, options=options)
    opt_time = round(time.time() - t0, 7)

    if check_grad(cost_fun, grad_bp, weights_0, args) >= 1e-3:
        print('Error in checking gradient!')

    weights = res["x"]
    W, V = reverse_params_col(weights, n_x, n_h, n_y)

    Training_error_reg = res["fun"]
    fin_norm = np.linalg.norm(res["jac"])
    mes = res['message']

    args = [X_Train, Y_Train, spread, 0, n_x, n_h, n_y]
    Training_error = cost_fun(weights, args)

    if validate:
        args = [X_val, Y_val, spread, 0, n_x, n_h, n_y]

        val_error = cost_fun(weights, args)
        results = {"W": W, "V": V, 'spread': spread, "Training_error_reg": Training_error_reg,
                   "Validation_error": val_error, 'n_x': n_x, 'n_h': n_h, 'n_y': n_y, "Training_error": Training_error,
                   'neurons': number_neurons, 'res': res}

    else:
        results = {"W": W, "V": V, 'spread': spread, "Training_error_0": Training_error_0,
                   "Training_error_reg": Training_error_reg, "Training_error": Training_error,
                   'n_x': n_x, 'n_h': n_h, 'n_y': n_y, 'neurons': number_neurons, 'res': res,
                   'init_norm': init_norm, 'fin_norm': fin_norm, 'mes': mes, 'time': opt_time}
    return results

def shallow_NN_predict(X_Test, Y_Test, results, test_er=True):
    W = results['W']
    V = results['V']
    prediction, _ = forward_propagation(X_Test, {'W': W, 'V': V}, results['spread'])
    if test_er:
        args = [X_Test, Y_Test, results['spread'], 0, results['n_x'], results['n_h'], results['n_y']]
        test_error = cost_fun(results['res'].x, args)
    else:
        test_error = None
    return prediction, test_error