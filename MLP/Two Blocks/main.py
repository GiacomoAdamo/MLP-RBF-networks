from utilities import  *

np.random.seed(28927)

# Import and read the data from DATA.csv file
data = pd.read_csv("DATA.csv")
data = data.values.astype('float32')
X = data[:, 0: 2].T
Y = data[:, 2].reshape(-1, 1).T
vettore_uni = np.ones(len(X[0])).reshape(1, -1)
X1 = np.append(X, vettore_uni, axis=0)

# Split data in training set and test set
X_Train = X1[:, :186]
Y_Train = Y[:, :186]
X_Test = X1[:, 186:]
Y_Test = Y[:, 186:]

# Define the hyperparameters
N = 45
spread = 1.6
ro = 10**-5

# Train the network
results = minimize(X_Train, Y_Train, N, spread, ro)

# Testing phase
V = results["V"]
W = results["W"]

args = [X_Test, Y_Test, W, spread]
pred = predict(V, args)

print(f"""
Number neurons N: {N}
Value of the spread σ: {spread}
Value of ρ: {ro}
Max iteration: No needed
Optimization solver: lsq_linear
Number of function evaluation: 0
Number of gradient evaluation: 0
Time for optimizing: {results['time']} seconds
Training error: {results["Training_non_regularized_error"]}
Test error: {pred[1]}""")
