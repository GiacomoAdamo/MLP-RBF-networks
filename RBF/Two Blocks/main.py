from utilities import *

np.random.seed(42)

# Import and read the data from DATA.csv file
data = pd.read_csv("DATA.csv")
data = data.values.astype('float32')
X = data[:, 0: 2].T
Y = data[:, 2].reshape(-1, 1).T

# Split data in training set and test set
X_Train = X[:, 0:186]
Y_Train = Y[:, 0:186]
X_Test = X[:, 186:]
Y_Test = Y[:, 186:]
P_train, P_test = X_Train.shape[1], X_Test.shape[1]

# Define the hyperparameters
spread = 0.9
N = 62
L2 = 0.00001

# Compute the centers as centroids
kmeans = KMeans(n_clusters=N, n_init=20).fit(X_Train.T)
C = kmeans.cluster_centers_.T
# Train the network
argTrain = [X_Train, Y_Train, spread, L2, P_train, N, C]
res, opt_time = minimize(argTrain)
argTrain = [X_Train, Y_Train, spread, 0, P_train, N, C]
# Testing phase
argTest = [X_Test, Y_Test, spread, P_test, N, C]
pred, TestError = predict(res.x, argTest)

print(f"""
Number neurons N: {N}
Value of the spread σ: {spread}
Value of ρ: {L2}
Max iteration: No needed
Optimization solver: lsq_linear
Number of function evaluation: 0
Number of gradient evaluation: 0
Time for optimizing: {opt_time} seconds
Training error: {costfun(res.x,argTrain)}
Test error: {TestError}""")

