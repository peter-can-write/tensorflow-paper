import tflearn
import tflearn.datasets.mnist as mnist

X, Y, validX, validY = mnist.load_data(one_hot=True)

# Building our neural network
input_layer = tflearn.input_data(shape=[None, 784])
output_layer = tflearn.fully_connected(input_layer, 10, activation='softmax')

# Optimization
sgd = tflearn.SGD(learning_rate=0.5)
net = tflearn.regression(output_layer, optimizer=sgd)

# Training
model = tflearn.DNN(net)
model.fit(X, Y, validation_set=(validX, validY))
