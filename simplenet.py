"""
Lets try to make a simple network
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy

from PIL import Image
from scipy import ndimage

import tensorflow as tf
from tensorflow.python.framework import ops

from cnn_utils import *


#   #########################################################################


def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    W2 = parameters['W2']

    # CONV2D: stride of 1, padding 'SAME'
    s = 1
    Z1 = tf.nn.conv2d(X, W1, strides=[1, s, s, 1], padding='SAME')
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    f, s = 8, 8
    P1 = tf.nn.max_pool(A1, ksize=[1, f, f, 1], strides=[1, s, s, 1], padding='SAME')
    # CONV2D: filters W2, stride 1, padding 'SAME'
    s = 1
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, s, s, 1], padding='SAME')
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    f, s = 4, 4
    P2 = tf.nn.max_pool(A2, ksize=[1, f, f, 1], strides=[1, s, s, 1], padding='SAME')
    # FLATTEN
    P2 = tf.contrib.layers.flatten(P2)
    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None"
    num_outputs = 4
    Z3 = tf.contrib.layers.fully_connected(P2, num_outputs, activation_fn=None)

    return Z3


#   #########################################################################


def simple_model(X_train, Y_train, X_test, Y_test, learning_rate=0.009,
                 num_epochs=100, minibatch_size=64, print_cost=True):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X_train -- training set, of shape (None, 64, 64, 3)
    Y_train -- test set, of shape (None, n_y = 6)
    X_test -- training set, of shape (None, 64, 64, 3)
    Y_test -- test set, of shape (None, n_y = 6)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)  # to keep results consistent (tensorflow seed)
    seed = 3  # to keep results consistent (numpy seed)
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []  # To keep track of the cost

    # Create Placeholders of the correct shape
    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32, [None, n_y])

    # Initialize parameters
    W1 = tf.get_variable("W1", [4, 4, 1, 8], initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer())

    parameters = {"W1": W1,
                  "W2": W2}

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters)

    # Cost function: Add cost function to tensorflow graph
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Initialize all the variables globally
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    print('Starting TensorFlow session...')
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                _, temp_cost = sess.run([optimizer, cost], {X: minibatch_X, Y: minibatch_Y})

                minibatch_cost += temp_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        return train_accuracy, test_accuracy, parameters


#   #########################################################################

#   TEST

#   #   #########################################################################
print('  --- START ---  ')

# Load data
trstep = 5

train_lists = {'gla': np.arange(1, 72, trstep),
               'epi': np.arange(175, 270, trstep),
               'pap': np.arange(310, 390, trstep),
               'der': np.arange(400, 700, trstep)}

# print(train_lists['gla'].shape)
# print(train_lists['epi'].shape)
# print(train_lists['pap'].shape)
# print(train_lists['der'].shape)
# input('Wait...')

label_lists = {'gla': np.array([[1, 0, 0, 0]]),
               'epi': np.array([[0, 1, 0, 0]]),
               'pap': np.array([[0, 0, 1, 0]]),
               'der': np.array([[0, 0, 0, 1]])}


X_train = np.empty((0, 240, 240, 1))
Y_train = np.empty((0, 4))
print(X_train.shape)
print(Y_train.shape)
for label in train_lists.keys():
    files = []
    for elem in train_lists[label]:
        fold = '/home/manuel/PycharmProjects/SimpleNet/data/'
        name = 'sm_Bicubic'
        numb = str(elem).zfill(4)
        exte = '.tif'
        file = fold + name + numb + exte
        files.append(np.array(Image.open(file)).reshape(240, 240, 1))
    X_train = np.vstack((X_train, np.array(files)))
    Y_train = np.vstack((Y_train, np.repeat(label_lists[label], train_lists[label].shape[0], 0)))

print('%i examples loaded' % X_train.shape[0])
print('X: ', X_train.shape)
print('Y: ', Y_train.shape)
X_test = X_train
Y_test = Y_train

input('wait...')

print('\nStarting the training')
_, _, parameters = simple_model(X_train, Y_train, X_test, Y_test)

