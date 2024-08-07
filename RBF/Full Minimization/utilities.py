import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize, check_grad

def gaussian(x,spread=1):
    return np.exp(-((x / spread) ** 2))

def matrixPhi(X, C, spread):
    # X is a matrix ov dimension (2*P) While c is a matrix of dimension (2*N)
    #matrix=np.sqrt(np.square(np.dot(X.T,-C)))
    #phi=gaussian(matrix,spread)
    #return phi #the final matrix phi has dimension (P*N)
    x1 = X[0].reshape(-1, 1)
    c1 = C[0].reshape(1, -1)
    x2 = X[1].reshape(-1, 1)
    c2 = C[1].reshape(1, -1)

    diff = np.square(x1 - c1) + np.square(x2 - c2)
    norms = np.sqrt(diff)
    Phi = gaussian(norms, spread)
    return Phi, diff

def params_col(parameters):
    param1 = parameters['W'].reshape((-1,1))
    param2 = parameters['C'].reshape((-1,1))
    param =  np.concatenate((param1,param2))
    return param

def reverse_params_col(param, N):
    V = param[0:N].reshape((N,1))
    C = param[N:].reshape((2,N))
    return V, C

def costfun(weights,argsfun):
    """argfun is a list containing [X,Y,spread,ro1,ro2,P,N]
    weights is a vector containing both c and w, so the final dimension
    of the weightS IS N + P*N"""
    X, Y = argsfun[0], argsfun[1]
    spread, ro1, ro2 = argsfun[2], argsfun[3], argsfun[4]
    P, N = argsfun[5], argsfun[6]
    V, C = reverse_params_col(weights, N)
    Phi_matrix, _ = matrixPhi(X,C,spread)
    reg_term = (ro1 / 2) * np.square(np.linalg.norm(V)) + (ro2 / 2) * np.square(np.linalg.norm(C))
    cost = (1 / (2 * P)) * np.square(np.linalg.norm(np.dot(Phi_matrix,V) - Y.T))
    cost = cost + reg_term
    return cost

def gradient(weights, argsfun):
    """argfun is a list containing [X,Y,spread,ro1,ro2,P,N]
    weights is a vector containing both c and w, so the final dimension
    of the weightS IS N + P*N"""
    X, Y = argsfun[0], argsfun[1]
    spread, ro1, ro2 = argsfun[2], argsfun[3], argsfun[4]
    P, N = argsfun[5], argsfun[6]
    V, C = reverse_params_col(weights, N)
    Phi_matrix, diff = matrixPhi(X, C, spread)  # dim (n_samples, n_neurons)
    E = np.dot(Phi_matrix, V).T - Y  # dim (n_out=1, n_samples)
    dV = (1 / P) * np.dot(E, Phi_matrix).T + ro1 * V
    grads = dV
    X_Aumentata = np.repeat(X[np.newaxis, :, :], N, axis=0)
    Q = (X_Aumentata.T - C).T  # mi calcolo le differenze x_i - c_j (dim_input, n_samples)
    QUBO = np.moveaxis(Q, 1, 2)
    grad_phi = (2 / (spread ** 2)) * np.multiply(np.exp(-((np.linalg.norm(QUBO, axis=2).T / spread) ** 2)),
                                                 np.moveaxis(Q, 0, 2))
    # dim (n_samples, 2)
    a = np.dot(Phi_matrix, V).T - Y
    dC = (V / P).T * np.sum(np.multiply(a.T, grad_phi), axis=1) + ro2 * C

    grads = np.concatenate((grads, dC.reshape(-1, 1)))
    return np.array(grads).reshape(-1)

def RBF_train(X_Train, Y_Train, X_val, Y_val, N, spread, ro1, ro2, max_iter, validate=True):
    P = X_Train.shape[1]

    weights_0 = np.random.randn(N + 2 * N)
    args = [X_Train, Y_Train, spread, 0, 0, P, N]
    Training_error_0 = costfun(weights_0, args)
    init_norm = np.linalg.norm(gradient(weights_0, args))

    args = [X_Train, Y_Train, spread, ro1, ro2, P, N]
    options = {'maxiter': max_iter}
    t0 = time.time()
    res = minimize(costfun, weights_0, args=args, method='BFGS', jac=gradient, options=options)
    opt_time = time.time() - t0

    if check_grad(costfun, gradient, weights_0, args) >= 1e-3:
        print('Error in checking gradient!')

    weights = res["x"]
    V, C = reverse_params_col(weights, N)

    Training_error_reg = res["fun"]
    fin_norm = np.linalg.norm(res["jac"])
    mes = res['message']

    args = [X_Train, Y_Train, spread, 0, 0, P, N]
    Training_error = costfun(weights, args)

    if validate:
        P = X_val.shape[1]
        args = [X_val, Y_val, spread, 0, 0, P, N]

        val_error = costfun(weights, args)
        results = {"V": V, "C": C, 'spread': spread, "Training_error_reg": Training_error_reg,
                   "Validation_error": val_error, 'neurons': N, 'res': res, "Training_error": Training_error}

    else:
        results = {"V": V, "C": C, 'spread': spread, "Training_error_0": Training_error_0,
                   "Training_error_reg": Training_error_reg, "Training_error": Training_error,
                   'neurons': N, 'res': res, 'init_norm': init_norm, 'fin_norm': fin_norm, 'mes': mes, 'time': opt_time}
    return results

def RBF_predict(X_Test, Y_Test, results, test_er = True):
    P = X_Test.shape[1]
    V, C = results['V'], results['C']
    spread, N = results['spread'], results['neurons']
    Phi_matrix, _ = matrixPhi(X_Test, C, spread)
    prediction = np.dot(Phi_matrix,V).reshape(-1)
    if test_er:
        args = [X_Test, Y_Test, spread, 0, 0, P, N]
        test_error = costfun(results['res'].x, args)
    else:
        test_error = None
    return prediction, test_error
