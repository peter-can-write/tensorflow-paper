#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" A one-hidden-layer-MLP MNIST-classifier. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import the training data (MNIST)
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

# Possibly download and extract the MNIST data set.
# Retrieve the labels as one-hot-encoded vectors.
mnist = input_data.read_data_sets("/tmp/mnist", one_hot=True)

# Create a new graph
graph = tf.Graph()

# Set our graph as the one to add nodes to
with graph.as_default():

    # Placeholder for input examples (None = variable dimension)
    examples = tf.placeholder(shape=[None, 784], dtype=tf.float32)
    # Placeholder for labels
    labels = tf.placeholder(shape=[None, 10], dtype=tf.float32)

    # Draw the weights from a random uniform distribution for symmetry breaking
    weights = tf.Variable(tf.random_uniform(shape=[784, 10]))
    # Slightly positive biases to avoid dead neurons
    bias = tf.Variable(tf.constant(0.1, shape=[10]))

    # Apply an affine transformation to the input features
    logits = tf.matmul(examples, weights) + bias
    estimates = tf.nn.softmax(logits)

    # Compute the cross-entropy
    cross_entropy = -tf.reduce_sum(labels * tf.log(estimates),
                                   reduction_indices=[1])
    # And finally the loss
    loss = tf.reduce_mean(cross_entropy)

    # Create a gradient-descent optimizer that minimizes the loss.
    # We choose a learning rate of 0.5
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Find the indices where the predictions were correct
    correct_predictions = tf.equal(
        tf.argmax(estimates, dimension=1),
        tf.argmax(labels, dimension=1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    for step in range(1000):
        example_batch, label_batch = mnist.train.next_batch(100)
        feed_dict = {examples: example_batch, labels: label_batch}
        if step % 100 == 0:
            _, loss_value, accuracy_value = session.run(
              [optimizer, loss, accuracy],
              feed_dict=feed_dict
            )
            print("Loss at time {0}: {1}".format(step, loss_value))
            print("Accuracy at time {0}: {1}".format(step, accuracy_value))
        else:
            optimizer.run(feed_dict)
