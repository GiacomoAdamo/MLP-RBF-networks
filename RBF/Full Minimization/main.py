from utilities import *

np.random.seed(1806447)

# Import and read the data from DATA.csv file
data = pd.read_csv("DATA.csv")
data = data.values.astype('float32')
X = data[:, 0: 2].T
Y = data[:, 2].reshape(-1, 1).T

# Split data in training set and test set
X_Train = X[:, :186]
Y_Train = Y[:, :186]
X_Test = X[:, 186:]
Y_Test = Y[:, 186:]

# Define the hyperparameters
spread, neurons, ro = 0.9, 62, 1
max_iter = 2000

# Train the network
best_results = RBF_train(X_Train, Y_Train, 0, 0, neurons, spread, ro*1e-5, ro*1e-5,  max_iter, False)

# Testing phase
_, test_error = RBF_predict(X_Test, Y_Test, best_results)


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