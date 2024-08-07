from utilities import *

# Ininizialize seed
np.random.seed(42)

# Import and read the data from DATA.csv file
data = pd.read_csv("DATA.csv")
data = data.values.astype('float32')
X = data[:, 0 : 2].T
Y = data[:, 2].reshape(-1, 1).T
vettore_uni = np.ones(len(X[0])).reshape(1, -1)
X1 = np.append(X, vettore_uni, axis=0)

# Split data in training set and test set
X_Train = X1[:, :186]
Y_Train = Y[:, :186]
X_Test = X1[:, 186:]
Y_Test = Y[:, 186:]

# Define the hyperparameters
max_iter = 5000
spread, neurons, ro = 1.6, 45, 1

# Train the network
best_results = shallow_NN_train(X_Train, Y_Train, 0, 0, neurons, spread, ro*10**-5, max_iter, False)
# Testing phase
_, test_error = shallow_NN_predict(X_Test, Y_Test, best_results)

print(f"""
Number neurons N: {neurons}
Value of the spread σ: {spread}
Value of ρ: {ro * 1e-5}
Max iteration: {max_iter}
Optimization solver: BFGS
Number of function evaluation: {best_results["res"].nfev}
Number of gradient evaluation: {best_results["res"].njev}
Time for optimizing: {best_results['time']} seconds
Training error: {best_results["Training_error"]}
Test error: {test_error}""")
