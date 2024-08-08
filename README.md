# MLP-RBF networks
In this project MultiLayer Perceptron (MLP) and Radial Basis Function (RBF) neural networks for 2D function regression are implemented from scratch. For the RBF a Gaussian Kernel was employed.

## Description

For each network, three different optimization techniques are tested:
 - Full Minimization
 - Two blocks method
 - Decomposition method

For each method a detailed explanation is provided in the sections below.

### Target problem


### Full Minimization 

The target problem is a highly non-linear and non-convex, therefore the goal of the optimization procedure is not to find the global minimizer of the regularized training error, but to find one of the local minimizers. In this project a batch method has been choosen rather than an online method.

In addition, to speed up the performance of the algorithm, we implemented the forward propagation and the back propagation in a vectorized form, using the python broadcasting and the native numpy function for handling matrices operations, without loops. 

The python routine used to solve the optimization problem is scipy.optimize.minimize employing as unconstraint first order derivative methods the BFGS one.

To enhance the performances of the method, it employs the callable function of the gradient so as not to have it estimated by the algorithm. The accuracy of our computation of the gradient is checked using the function check grad leading to a difference between the gradient computed by us and the one evaluated with finite differences of the order of magnitude of $10^−6.

### Two blocks method

Once 𝑊 (the first layer parameter) is fixed, the problem of minimizing the regularized training error is a quadratic convex problem with a unique global
minimizer. The problem can be reformulated in the form of 1
2𝑃 ‖𝐴𝑉 − 𝐵‖2
, where 𝑉 is the 𝑁 × 1 vector of the output weights,
while 𝐴 = ((𝑔(𝑊𝑋 + 𝑏)𝑇
√𝜌𝑃𝐼𝑁
) 𝑎𝑛𝑑 𝐵 = ( 𝑌
0𝑁
).
To solve this problem, we first constructed the matrices 𝐴 and 𝐵. Then we used the solver lsq_linear from scipy.optimize. This
method requires as mandatory parameters 𝐴, 𝐵. Concerning the other parameters, we used the default values. This solver finds
the global minimizer of the linear least square problem with 0 iterations.

## Getting Started

### Dependencies and Executing program

 - Download the repo with all the subdirectiories of a subset of interest
 - Check if Python 3.10 is installed on your machine 
 - Run the following command to install al the dependencies required for this project
```
pip install -r requirements.txt
```
- To run a specific network of optimization technique, launch the command
```
python main.py
```
## Data 

### Data format

Here, input data are not provided but is important to pass as input a CSV file with the following format:

 - X1 : input variable first dimension 
 - X2 : input variable second dimension
 - Y  : output variable


## Authors

Author name and contact info

- [Giacomo Mattia Adamo](www.linkedin.com/in/giacomo-mattia-adamo-b36a831ba)

## License

This project is licensed - see the LICENSE.md file for details
