import tensorflow as tf
import numpy as np

# y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype(np.float32)
y = x_data * 0.1 + 0.3

# Create new computational graph
graph = tf.Graph()

# Register the graph as the default
with graph.as_default():

    # Weights and bias
    W = tf.Variable(0.0)
    b = tf.Variable(0.0)

    # Produce our estimate y_hat
    y_hat = W * x_data + b

    # Minimize the mean squared error between
    # our estimate y_hat and the label y
    loss = tf.reduce_mean(tf.square(y_hat - y))

    # Perform gradient descent
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# Enter session environment
with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    # Train our algorithm!
    for step in range(100):
        _, W_value, b_value = session.run([optimizer, W, b])

        print('W: {0:.3f} b: {1:.3f}'.format(W_value, b_value))
