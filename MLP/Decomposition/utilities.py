import numpy as np
import time
import pandas as pd
from scipy.optimize import minimize
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


def derivata_tanh(x, spread=1):
    # (4 E^(2 s x) s)/(1 + E^(2 s x))^2
    return (4 * spread * np.exp(2 * spread * x)) / ((np.exp(2 * spread * x) + 1) ** 2)


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
    A1 = np.dot(W, X)  # HO CERCATO DI USARE LA SIMBOLOGIA DELLA PALAGI: A1 è LA COMBINAZIONE LINEARE
    Z1 = tanh(A1, spread)  # Z1 è A SEGUITO DELL'ATTIVAZIONE
    A2 = np.dot(V, Z1)  # NELL'OUPUT LAYER NON HO MESSO IL BIAS
    Z2 = A2

    cache = {"A1": A1, "Z1": Z1, "A2": A2, "Z2": Z2}

    return Z2, cache


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


def cost_function2(weights,argsfun): #this function is the objective function of the second block of the 2 block decomposition
    #weights is a column vector containing just W, so the dimension is n_h*3
    #argsfun is a list containing [X,Y,V,spread,L2]
    X,Y = argsfun[0],argsfun[1]
    V,spread = argsfun[2],argsfun[3]
    L2=argsfun[4]
    W=weights.reshape(-1,3)
    V=V.reshape(1,-1)
    Z1=compute_Z1(X,W,spread)
    P=len(Y[0])
    Z2 = np.dot(V, Z1)
    cost = (1 / (2 * P))* np.square(np.linalg.norm(Z2 - Y)) + (L2 / 2) * np.square(np.linalg.norm(weights))
    return cost

def grad_cost_function2(weights,argsfun): #this function computes the gradient with respect to W, which is the only variable in the second part of the 2 block decomposition
    #weights is a column vector containing just W, so the dimension is n_h*3
    #argsfun is a list containing [X,Y,V,spread,L2]
    X,Y = argsfun[0],argsfun[1]
    V,spread = argsfun[2],argsfun[3]
    L2=argsfun[4]
    W=weights.reshape(-1,3)
    V=V.reshape(1,-1)
    A1 = np.dot(W, X)
    Z1 = tanh(A1, spread)
    P=len(Y[0])
    Z2 = np.dot(V, Z1)
    E = Z2 - Y
    dV = (1 / P) * np.dot(E, Z1.T) + L2 * V
    delta = np.multiply(derivata_tanh(A1, spread), (np.dot(V.T, E)))
    dW = (1 / P) * np.dot(delta, X.T) + L2 * W
    return np.squeeze(dW.reshape(-1,1))


def Minimize(X_Train, Y_Train, X_val, Y_val, number_neurons, spread, L2, patience=20):
    bound = 10 ** 6
    counter = 0
    n_x, n_h, n_y = layer_sizes(X_Train, Y_Train, number_neurons)
    matrix_b = np.append(Y_Train.T, np.zeros(n_h).reshape(-1, 1), axis=0).reshape(-1)
    P = len(Y_Train[0])
    lambda_tilde = np.sqrt(L2 * P)
    d = {}
    W = np.random.randn(n_h * n_x).reshape(n_h,
                                           n_x)  # np.random.normal(0.15996195977682795, 0.5544182407508429, n_h * n_x).reshape(n_h, n_x)  #np.random.randn(n_h * n_x).reshape(n_h, n_x) #first, we initialize W as a random matrix
    t0 = time.time()
    for k in range(5000):
        if k < 10:
            gtol = 0.01
            maxit = k
        else:
            if k < 50:
                gtol = 0.001
                maxit = k
            else:
                gtol = 1e-5
                maxit = k
        Z1 = compute_Z1(X_Train, W, spread).T
        matrix_A = np.append(Z1, lambda_tilde * np.identity(n_h), axis=0)
        res = lsq_linear(matrix_A, matrix_b, lsmr_tol='auto',
                         verbose=0)  # we determine V solving a linear least square problem
        V = res["x"]
        d["V"] = V

        # argsfun is a list containing [X,Y,V,spread,L2]
        argsfun = [X_Train, Y_Train, V, spread, L2]
        options = {'gtol': gtol, "maxiter": maxit}
        results = minimize(cost_function2, W.reshape(-1), method='bfgs',
                           args=argsfun, jac=grad_cost_function2, options=options)
        W = results["x"].reshape(-1, 3)
        d["W" + str(k)] = W.reshape(-1, 3)

        parameters = [X_Train, Y_Train, W.reshape(-1, 3), spread, 0]
        training_error = cost_function(V, parameters)
        d["training_error" + str(k)] = training_error

        parameters = [X_val, Y_val, W.reshape(-1, 3), spread, 0]  # in order to calculate error without regolarization
        validation_error = cost_function(V, parameters)
        d["validation_error" + str(k)] = validation_error
        if k >= 1:
            if validation_error <= bound:
                counter = 0
                bound = validation_error
            else:
                counter += 1
                if counter > patience:
                    opt_time = time.time() - t0
                    final_resultss = {"iterations": k, "training error": d["training_error" + str(k - patience)],
                                      "validation error": d["validation_error" + str(k - patience)],
                                      "W": d["W" + str(k - patience)], "V": d["V"], 'time': opt_time}
                    break

    return final_resultss


def full_Minimization(X_Train, Y_Train, spread, L2, results, number_neurons):
    n_x, n_h, n_y = layer_sizes(X_Train, Y_Train, number_neurons)
    matrix_b = np.append(Y_Train.T, np.zeros(n_h).reshape(-1, 1), axis=0).reshape(-1)
    P = len(Y_Train[0])
    lambda_tilde = np.sqrt(L2 * P)
    d = {}
    W = np.random.randn(n_h * n_x).reshape(n_h, n_x)  # first, we initialize W as a random matrix
    t0 = time.time()
    gev = 0
    fev = 0
    for k in range(results["iterations"] - 20):
        if k < 10:
            gtol = 0.01
            maxit = k
        else:
            if k < 50:
                gtol = 0.001
                maxit = k
            else:
                gtol = 1e-5
                maxit = k
        Z1 = compute_Z1(X_Train, W, spread).T
        matrix_A = np.append(Z1, lambda_tilde * np.identity(n_h), axis=0)
        res = lsq_linear(matrix_A, matrix_b, lsmr_tol='auto',
                         verbose=0)  # we determine V solving a linear least square problem
        V = res["x"]
        d["V"] = V

        # argsfun is a list containing [X,Y,V,spread,L2]
        argsfun = [X_Train, Y_Train, V, spread, L2]
        options = {'gtol': gtol, "maxiter": maxit}
        results = minimize(cost_function2, np.squeeze(W.reshape(-1, 1)), method='bfgs',
                           args=argsfun, jac=grad_cost_function2, options=options)
        W = results["x"].reshape(-1, 3)
        fev += results.nfev
        gev += results.njev
        d["W" + str(k)] = W.reshape(-1, 3)

        parameters = [X_Train, Y_Train, W.reshape(-1, 3), spread, 0]
        training_error = cost_function(V, parameters)
        d["training_error" + str(k)] = training_error

    opt_time = time.time() - t0
    final_resultss = {"iterations": k, "training error": d["training_error" + str(k)],
                      "W": d["W" + str(k)], "V": d["V"], 'time': opt_time, "fev": fev, "gev": gev}
    return final_resultss


def predict(V,W,args, err=True):
    #V is the only vector of variables
    #args is a list containing[Xtest,Ytest,spread]
    X, Y = args[0], args[1]
    spread = args[2]
    Z1 = compute_Z1(X, W, spread)
    y_prediction = np.dot(V.T, Z1)
    if err:
        parameters = [X, Y, W, spread, 0]
        test_error = cost_function(V, parameters)
    else:
        test_error = None
    return y_prediction, test_error







