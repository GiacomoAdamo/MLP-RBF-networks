import time
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from sklearn.cluster import KMeans

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
    return Phi

def phi_tilde(phi, L2, N, P):
    Phi_tilde = np.concatenate((phi,np.sqrt(P*L2)*np.identity(N)))
    return Phi_tilde

def y_tilde(Y, N, P):
    Y_tilde = np.concatenate((Y.T, np.zeros((N, 1))))
    return Y_tilde.reshape(P+N,1)

def costfun(V, argsfun):
    """argfun is a list containing [X,Y,spread,L2,P,N,C]"""
    X, Y = argsfun[0], argsfun[1]
    spread, L2 = argsfun[2], argsfun[3]
    P, N, C = argsfun[4], argsfun[5], argsfun[6]
    V = V.reshape(N, 1)
    Phi_matrix = matrixPhi(X, C, spread)

    Phi_tilde = phi_tilde(Phi_matrix, L2, N, P)
    Y_tilde = y_tilde(Y, N, P)
    reg_term = (L2 / 2) * np.square(np.linalg.norm(V))
    cost = (1 / (2 * P)) * np.square(np.linalg.norm(np.dot(Phi_matrix, V).reshape(-1, 1) - Y.T))
    cost = cost + reg_term
    return np.squeeze(cost)

def gradient(V, argsfun):
    """argfun is a list containing [X,Y,spread,L2,P,N,C]"""
    X, Y = argsfun[0], argsfun[1]
    spread, L2 = argsfun[2], argsfun[3]
    P, N, C = argsfun[4], argsfun[5], argsfun[6]

    V = V.reshape(N, 1)
    Phi_matrix = matrixPhi(X, C, spread)

    Phi_tilde = phi_tilde(Phi_matrix, L2, N, P)
    Y_tilde = y_tilde(Y, N, P)
    a = np.dot(Phi_tilde.T, Phi_tilde)
    grad = np.dot(a, V) - np.dot(Phi_tilde.T, Y_tilde).reshape(N, 1)
    grad = grad / P

    return grad.reshape(-1)

def minimize(argsfun):
    """argfun is a list containing [X,Y,spread,L2,P,N,C]"""
    X, Y = argsfun[0], argsfun[1]
    spread, L2 = argsfun[2], argsfun[3]
    P, N, C = argsfun[4], argsfun[5], argsfun[6]

    Phi_matrix = matrixPhi(X, C, spread)

    Phi_tilde = phi_tilde(Phi_matrix, L2, N, P)
    Y_tilde = y_tilde(Y, N, P)
    A = Phi_tilde
    b = Y_tilde.reshape(-1)

    t0 = time.time()
    res = lsq_linear(A, b, lsmr_tol='auto', verbose=0)
    opt_time = time.time() - t0
    return res, round(opt_time, 5)

def predict(V, argsfun, error=True):
    """argfun is a list containing [X,Y,spread,P,N,C]"""
    X, Y = argsfun[0], argsfun[1]
    spread = argsfun[2]
    P, N, C = argsfun[3], argsfun[4], argsfun[5]

    Phi_matrix = matrixPhi(X, C, spread)

    predictions = np.dot(Phi_matrix, V).reshape(-1, 1)

    if error:
        TestError = (1 / (2 * P)) * np.square(np.linalg.norm(np.dot(Phi_matrix, V).reshape(-1, 1) - Y.T))
    else:
        TestError = None
    return predictions, TestError

