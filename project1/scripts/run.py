# Useful starting lines
import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from proj1_helpers import *
from costs import *


DATA_TRAIN_PATH = '../Data/train.csv' # TODO: add a file Data-Project1 with the train data 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

# Delete the outliers with the median
def delete_outliers(tX):
    for idx_feature in range(tX.shape[1]):
        tX_feature = tX[:,idx_feature]
        median = np.median(tX_feature[np.where(tX_feature != -999)])
        new = np.where(tX_feature == -999, median, tX_feature)
        tX[:, idx_feature] = np.copy(new)
    return tX

tX = delete_outliers(tX)

# Standardize the data
stx, mean_stx, std_x = standardize(tX)

# y must be 0 or 1 and not -1 or 1
def set_y(y):
    y = np.where(y == -1, 0, y)
    return y
y = set_y(y)

'''
def calculate_correlation(stx):
    corr = np.ones((stx.shape[1]-1, stx.shape[1]-1))
    for feature1 in range(1, stx.shape[1]):
        for feature2 in range(1, stx.shape[1]):
            corr[feature1-1, feature2-1] = np.corrcoef(stx[:, feature1], stx[:, feature2])[0, 1]
            if (corr[feature1-1, feature2-1] >= 0.9 and feature1-1 != feature2-1):
                
                print("Features {f1} and {f2} are highly correlated: {corr}".format(f1 =feature1-1, f2 = feature2-1, corr = corr[feature1-1, feature2-1]))
    return corr

corr = calculate_correlation(stx)
'''
idx_to_del = np.array([22, 30])

'''
def calculate_correlation_with_y(stx, y, threshold):
    corr = np.ones(stx.shape[1]-1)
    for feature in range(1, stx.shape[1]):
        corr[feature-1] = np.corrcoef(y, stx[:, feature])[0, 1] 
        if (abs(corr[feature-1]) <= threshold):
            print("feature {f} is not correlated with y: {corr}".format(f = feature-1, corr= corr[feature-1]))
    return corr

corr = calculate_correlation_with_y(stx, y, 0.005)
'''

idx_to_del = np.append(idx_to_del, [ 15, 16, 18, 19, 25, 26, 28, 29])

def delete_features(stx):
    return np.delete(stx, idx_to_del, 1)

clean_tx = delete_features(np.copy(stx))


# Use a polynomial basis
def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    dim = degree+1
    N = x.shape[0]
    phi = np.ones((N, x.shape[1]))
    for j in range(1, dim):
        phi = np.concatenate((phi, np.power(x,j)), axis =1)
    return phi

#Build degree 3 polynomial basis
degree = 3
poly_tx = build_poly(clean_tx, degree)
poly_tx.shape


# Implement regularized logistic regression on the degree 3 polynomial basis
def sigmoid(t):
    """apply sigmoid function on t."""
    # equivalent to use 1/(1+exp(-t)) but avoids overflow
    return np.exp(-np.logaddexp(0, -t))

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    #for n in range(N):
    #   cost += np.log(1+np.exp(tx[n, :].T @ w)) - (y[n] * tx[n, :].T @ w)
    y = y.reshape((-1, 1))
    return np.sum(np.logaddexp(0, tx @ w)) - y.T @ (tx @ w)

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    sig = sigmoid(tx @ w)
    sig = sig.reshape(sig.shape[0],)
    return tx.T @ (sig - y)

def learning_by_gradient_descent(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y, tx, w) + lambda_* (w.T @ w)
    gradient = calculate_gradient(y, tx, w)
    
    w.shape = (w.shape[0],)
    w = w - gamma * gradient
    return loss, w


def reg_logistic_regression(y, tx, lambda_, gamma, max_iters):
    """
    Logistic regression using GD
    """
    # init parameters
    threshold = 1e-4
    losses = []

    # build w
    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma, lambda_)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # converge criteria ( max_iters is really high)
        losses.append(np.copy(loss))
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return loss, w


#Define parameters and run code
gamma = 1e-7
max_iters = 10000
lambda_ = 0.07
loss, w = reg_logistic_regression(y, poly_tx, lambda_, gamma, max_iters)

DATA_TEST_PATH = '../Data/test.csv' 
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

# Apply same pre-processing on test data
tX_test = delete_outliers(tX_test)
stx_test, mean_stx_test, std_x_test = standardize(tX_test)
clean_test = delete_features(stx_test)
poly_test = build_poly(clean_test, degree)

#Create CSV
OUTPUT_PATH = '../Data/Submission( Group 20).csv' 
y_pred = predict_labels(w, poly_test)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)


