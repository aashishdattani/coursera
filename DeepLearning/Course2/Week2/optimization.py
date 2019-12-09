import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

from DeepLearning.Course2.Week2.util.opt_utils_v1a import load_params_and_grads, initialize_parameters, forward_propagation, backward_propagation
from DeepLearning.Course2.Week2.util.opt_utils_v1a import compute_cost, predict, predict_dec, plot_decision_boundary, load_dataset
from DeepLearning.Course2.Week2.util.testCases import *


def update_parameters_with_gd(parameters, grads, learning_rate):
    L = len(parameters) // 2  # number of layers in the neural networks

    # Update rule for each parameter
    for l in range(L):
        parameters["W" + str(l + 1)] -= (learning_rate * grads['dW' + str(l + 1)])
        parameters["b" + str(l + 1)] -= (learning_rate * grads['db' + str(l + 1)])

    return parameters


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    np.random.seed(seed)  # To make your "random" minibatches the same as ours
    m = X.shape[1]  # number of training examples
    mini_batches = []

    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, (k + 1) * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[:, (k + 1) * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


# GRADED FUNCTION: initialize_velocity

def initialize_velocity(parameters):
    L = len(parameters) // 2  # number of layers in the neural networks
    v = {}

    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
        v["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)

    return v


def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    L = len(parameters) // 2  # number of layers in the neural networks

    for l in range(L):
        # compute velocities
        v["dW" + str(l + 1)] = (beta * v["dW" + str(l + 1)]) + ((1 - beta) * grads['dW' + str(l + 1)])
        v["db" + str(l + 1)] = (beta * v["db" + str(l + 1)]) + ((1 - beta) * grads['db' + str(l + 1)])
        # update parameters
        parameters["W" + str(l + 1)] -= (learning_rate * v["dW" + str(l + 1)])
        parameters["b" + str(l + 1)] -= (learning_rate * v["db" + str(l + 1)])

    return parameters, v


# GRADED FUNCTION: initialize_adam

def initialize_adam(parameters):
    L = len(parameters) // 2  # number of layers in the neural networks
    v = {}
    s = {}

    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
        v["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)
        s["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
        s["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)

    return v, s


# GRADED FUNCTION: update_parameters_with_adam

def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01,
                                beta1=0.9, beta2=0.999, epsilon=1e-8):
    L = len(parameters) // 2  # number of layers in the neural networks
    v_corrected = {}  # Initializing first moment estimate, python dictionary
    s_corrected = {}  # Initializing second moment estimate, python dictionary

    # Perform Adam update on all parameters
    for l in range(L):
        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        v["dW" + str(l + 1)] = (beta1 * v["dW" + str(l + 1)]) + ((1 - beta1) * grads['dW' + str(l + 1)])
        v["db" + str(l + 1)] = (beta1 * v["db" + str(l + 1)]) + ((1 - beta1) * grads['db' + str(l + 1)])

        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - (beta1 ** t))
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - (beta1 ** t))

        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        s["dW" + str(l + 1)] = (beta2 * s["dW" + str(l + 1)]) + (
                    (1 - beta2) * grads['dW' + str(l + 1)] * grads['dW' + str(l + 1)])
        s["db" + str(l + 1)] = (beta2 * s["db" + str(l + 1)]) + (
                    (1 - beta2) * grads['db' + str(l + 1)] * grads['db' + str(l + 1)])

        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - (beta2 ** t))
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - (beta2 ** t))

        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
        parameters["W" + str(l + 1)] -= (learning_rate * v_corrected["dW" + str(l + 1)] / (
                    (s_corrected["dW" + str(l + 1)] ** 0.5) + epsilon))
        parameters["b" + str(l + 1)] -= (learning_rate * v_corrected["db" + str(l + 1)] / (
                    (s_corrected["db" + str(l + 1)] ** 0.5) + epsilon))

    return parameters, v, s


def model(X, Y, layers_dims, optimizer, learning_rate=0.0007, mini_batch_size=64, beta=0.9,
          beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=10000, print_cost=True):
    L = len(layers_dims)  # number of layers in the neural networks
    costs = []  # to keep track of the cost
    t = 0  # initializing the counter required for Adam update
    seed = 10  # For grading purposes, so that your "random" minibatches are the same as ours
    m = X.shape[1]  # number of training examples

    # Initialize parameters
    parameters = initialize_parameters(layers_dims)

    # Initialize the optimizer
    if optimizer == "gd":
        pass  # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    # Optimization loop
    for i in range(num_epochs):

        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total = 0

        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation
            a3, caches = forward_propagation(minibatch_X, parameters)

            # Compute cost and add to the cost total
            cost_total += compute_cost(a3, minibatch_Y)

            # Backward propagation
            grads = backward_propagation(minibatch_X, minibatch_Y, caches)

            # Update parameters
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1  # Adam counter
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s,
                                                               t, learning_rate, beta1, beta2, epsilon)
        cost_avg = cost_total / m

        # Print the cost every 1000 epoch
        if print_cost and i % 1000 == 0:
            print("Cost after epoch %i: %f" % (i, cost_avg))
        if print_cost and i % 100 == 0:
            costs.append(cost_avg)

    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters


if __name__ == "__main__":
    train_X, train_Y = load_dataset()
    layers_dims = [train_X.shape[0], 5, 2, 1]
    parameters = model(train_X, train_Y, layers_dims, optimizer="adam")

    # Predict
    predictions = predict(train_X, train_Y, parameters)

    # Plot decision boundary
    plt.title("Model with Adam optimization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 2.5])
    axes.set_ylim([-1, 1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)