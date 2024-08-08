from utilities import *

np.random.seed(42)

# Import and read the data from DATA.csv file
data = pd.read_csv("DATA.csv")
data = data.values.astype('float32')
X = data[:, 0 : 2].T
Y = data[:, 2].reshape(-1, 1).T

vettore_uni = np.ones(len(X[0])).reshape(1, -1)
X1 = np.append(X, vettore_uni, axis=0)

# Split data in training set, validation set and test set
X_Train = X1[:, :150]
Y_Train = Y[:, :150]
X_Val = X1[:, 150:186]
Y_Val = Y[:, 150:186]
X_Train_true = X1[:, :186]
Y_Train_true = Y[:, :186]
X_Test = X1[:, 186:]
Y_Test = Y[:, 186:]

# Define the hyperparameters
neurons, spread, ro = 45, 1.6, 10**-5

# Train the network
results = Minimize(X_Train, Y_Train, X_Val, Y_Val,
                   neurons, spread, ro)
full_results = full_Minimization(X_Train_true, Y_Train_true,
                            1.6, 10**-5, results, 45)
V = full_results["V"]
W = full_results["W"].reshape(-1, 3)

# Testing phase
args = [X_Test, Y_Test, spread]
pred = predict(V, W, args)

print(f"""
Number neurons N: {neurons}
Value of the spread σ: {spread}
Value of ρ: {ro}
Max iteration of the two blocks:: 5000
Regarding the W problem:
    If number iteration (k) < 10: tolerance 0.01 and k max iterations 
    If number iteration (k) > 10 and < 50: tolerance 0.001 and k max iterations
    If number iteration (k) >= 50: tolerance 1e-5 and k max iterations
    Patience: 20 iterations 
Optimization solver: default lsq_solver for the V problem; BFGS for the W one
Total number of function evaluation (W problem): {full_results["fev"]}
Total umber of gradient evaluation (W problem): {full_results["gev"]}
Time for optimizing: {full_results['time']} seconds
Training error: {full_results["training error"]}
Test error: {pred[1]}
""")

