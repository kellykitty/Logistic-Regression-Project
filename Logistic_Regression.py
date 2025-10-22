# ================================
# Logistic Regression (from scratch) — Skeleton
# ================================

#imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#load data
def load_data():
    # load csv and convert yes/no to ones and zeros
    data = pd.read_csv("/Users/kelly/Downloads/CollegePlacement.csv")
    data["Placement"] = data["Placement"].map({"Yes": 1, "No": 0})
    data["Internship_Experience"] = data["Internship_Experience"].map({"Yes": 1, "No": 0})

    # use academic performance to predict placement
    X = data[["Academic_Performance"]].values
    y = data["Placement"].values

    # standardize features for clarity
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # split into training and testing
    split = int(0.8 * len(X))
    return X[:split], y[:split], X[split:], y[split:]

# load and prepare data
X, y, X_test, y_test = load_data()
m = X.shape[0]

# add the bias term
X = np.column_stack([np.ones(m), X])
n_with_bias = X.shape[1]

# sigmoid, loss, gradient, prediction fuctions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_proba(X, w):
    return sigmoid(np.dot(X, w))

def binary_cross_entropy(y_true, y_prob, eps=1e-12):
    # compute cost
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))

def gradient(X, y_true, y_prob):
    # compute the gradient for weight updates
    return (1 / len(y_true)) * np.dot(X.T, (y_prob - y_true))

# initialize paremeters
w = np.zeros(n_with_bias)  # initialize weights to zeros

# hyper pareameters 
learning_rate = 0.1        
num_iterations = 1000    
cost_history = []      

# gradient decent looping
for i in range(num_iterations):
    y_prob = predict_proba(X, w)         
    loss = binary_cross_entropy(y, y_prob)
    grad = gradient(X, y, y_prob)         
    w -= learning_rate * grad           
    cost_history.append(loss)   

# final paremeters
print("Final parameters (w):")
print(w)

# plot the cost versus iteration
plt.figure()
plt.plot(range(len(cost_history)), cost_history)
plt.xlabel("Iteration")
plt.ylabel("Cost (Log-Loss)")
plt.title("Training: Cost vs. Iterations")
plt.grid(True)
plt.show()

# plot cost vs 3 of the most important parameters
param_indices = np.argsort(np.abs(w[1:]))[::-1][:3] + 1  # top 3 non-bias weights
print("Plotting cost sensitivity for parameter indices:", param_indices)

def compute_cost_given_w(mod_w):
    # figure out cost for a changed weight vector
    y_hat_mod = predict_proba(X, mod_w)
    return binary_cross_entropy(y, y_hat_mod)

# clean selected parameters around trained values, see the sensitivity
for idx in param_indices:
    center = w[idx]
    sweep = np.linspace(center - 1.0, center + 1.0, 60)
    costs = []
    for val in sweep:
        w_tmp = w.copy()
        w_tmp[idx] = val
        costs.append(compute_cost_given_w(w_tmp))
    plt.figure()
    plt.plot(sweep, costs)
    plt.xlabel(f"Parameter w[{idx}]")
    plt.ylabel("Cost (Log-Loss)")
    plt.title(f"Cost vs Parameter w[{idx}] (holding others fixed)")
    plt.grid(True)
    plt.show()