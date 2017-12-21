"""
Lets try to make a simple network
"""

import datetime, os, sys

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
    W3 = parameters['W3']
    W4 = parameters['W4']

# 1
    # CONV2D: stride of 1, padding 'SAME'
    s = 1
    Z1 = tf.nn.conv2d(X, W1, strides=[1, s, s, 1], padding='SAME')
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    f, s = 8, 8
    P1 = tf.nn.max_pool(A1, ksize=[1, f, f, 1], strides=[1, s, s, 1], padding='SAME')

# 2
    # CONV2D: filters W2, stride 1, padding 'SAME'
    s = 1
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, s, s, 1], padding='SAME')
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    f, s = 4, 4
    P2 = tf.nn.max_pool(A2, ksize=[1, f, f, 1], strides=[1, s, s, 1], padding='SAME')

# 3
    # CONV2D: filters W2, stride 1, padding 'SAME'
    s = 1
    Z3 = tf.nn.conv2d(P2, W3, strides=[1, s, s, 1], padding='SAME')
    # RELU
    A3 = tf.nn.relu(Z3)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    f, s = 4, 4
    P3 = tf.nn.max_pool(A3, ksize=[1, f, f, 1], strides=[1, s, s, 1], padding='SAME')

# 4
    # CONV2D: filters W2, stride 1, padding 'SAME'
    s = 1
    Z4 = tf.nn.conv2d(P3, W4, strides=[1, s, s, 1], padding='SAME')
    # RELU
    A4 = tf.nn.relu(Z4)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    f, s = 4, 4
    P4 = tf.nn.max_pool(A4, ksize=[1, f, f, 1], strides=[1, s, s, 1], padding='SAME')


    # FLATTEN
    P4 = tf.contrib.layers.flatten(P4)
    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None"
    num_outputs = 4
    Z5 = tf.contrib.layers.fully_connected(P4, num_outputs, activation_fn=None)

    return Z5


#   #########################################################################


def simple_model(X_train, Y_train, X_test, Y_test, learning_rate=0.009,
                 num_epochs=100, minibatch_size=64, print_cost=True, restore_file=None):
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
    restore -- File path to restore variables from previous session

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
    W1 = tf.get_variable("W1", [3, 3, 1, 16], initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable("W2", [5, 5, 16, 32], initializer=tf.contrib.layers.xavier_initializer())
    W3 = tf.get_variable("W3", [3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
    W4 = tf.get_variable("W4", [5, 5, 64, 128], initializer=tf.contrib.layers.xavier_initializer())

    parameters = {"W1": W1,
                  "W2": W2,
                  "W3": W3,
                  "W4": W4}

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z5 = forward_propagation(X, parameters)

    # Cost function: Add cost function to tensorflow graph
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z5, labels=Y))

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Accuracy
    acc = tf.equal(tf.argmax(Z5, 1), tf.argmax(Y, 1))
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))

    # Initialize all the variables globally
    init = tf.global_variables_initializer()

    # Allow saving
    saver = tf.train.Saver(max_to_keep=5,
                           keep_checkpoint_every_n_hours=10000.0)

    # Start the session to compute the tensorflow graph
    print('Starting TensorFlow session...')
    with tf.Session() as sess:

        if restore_file is None:
            # Run the initialization
            print("Initializing parameters...")
            sess.run(init)
        else:
            # Restore variables from disk
            print("Restoring parameters...")
            saver.restore(sess, restore_file)

        # Define saving folder
        folder_name = 'saver_' \
                      + str(datetime.datetime.now().strftime("%y%m%d_%H%M")) \
                      + '_lr' + str(learning_rate) \
                      + '_ep' + str(num_epochs) \
                      + '_mb' + str(minibatch_size)
        print("Progress will be saved under ./%s/" % folder_name)

        # Tensorboard
        logs_path = './tmp/tf_logs/'
        # Create a summary to monitor cost tensor
        tf.summary.scalar("loss", cost)
        # Create a summary to monitor accuracy tensor
        tf.summary.scalar("accuracy", acc)
        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()
        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        # Do the training loop
        print("\n --- TRAINING --- ")
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            mb = 0
            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost. feedict should contain a minibatch for (X,Y).
                # + Run optimization op (backprop), cost op (loss) and summary nodes
                _, temp_cost, summary = sess.run([optimizer, cost, merged_summary_op],
                                                 {X: minibatch_X, Y: minibatch_Y})

                # Compute average loss
                minibatch_cost += temp_cost / num_minibatches

                # Write logs at every iteration
                summary_writer.add_summary(summary, epoch * len(minibatches) + mb)
                mb += 1

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)

            # Save the variables to disk every epoch.
            file_name = "epoch" #+ str(epoch).zfill(4) #+ ".ckpt"
            save_path = saver.save(sess, folder_name + '/' + file_name, global_step=epoch)
            # saver = tf.train.Saver(var_list=None)
            # saver.save(sess, file)
            print("Epoch " + str(epoch) + " saved in file: %s" % save_path)

        print('\nFINAL COST after %i epochs: ' % num_epochs, costs[-1])

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z5, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
        print("  --- DONE ---  \n")

        return train_accuracy, test_accuracy, parameters


#   #########################################################################

#   TEST

#   #   #########################################################################

