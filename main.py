from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow import keras as ks
from sklearn.preprocessing import MinMaxScaler

# load the data from the dataset
(training_images, training_labels), (test_images, test_labels) = ks.datasets.fashion_mnist.load_data()
test_labels = test_labels.astype(int)
batch_size = len(training_images)
scaler = MinMaxScaler()
print('Training Images Dataset Shape: {}'.format(training_images.shape))
print('No. of Training Images Dataset Labels: {}'.format(len(training_labels)))
print('Test Images Dataset Shape: {}'.format(test_images.shape))
print('No. of Test Images Dataset Labels: {}'.format(len(test_labels)))

# normalize the data for better training
# training_images = scaler.fit_transform(training_images.astype(np.float64))
# test_images = scaler.fit_transform(test_images.astype(np.float64))
training_images = training_images / 255.0
test_images = test_images / 255.0

# input layer
input_data_shape = (28, 28)
hidden_activation_function = 'relu'
output_activation_function = 'softmax'
dnn_model = ks.models.Sequential()

# convolutional and pooling layers
dnn_model.add(ks.layers.Flatten(input_shape=input_data_shape, name='Input_layer'))
dnn_model.add(ks.layers.Dense(256, activation=hidden_activation_function, name='Hidden_layer_1'))
## first Layer
pool1 = tf.compat.v1.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
dnn_model.add(ks.layers.Dense(192, activation=hidden_activation_function, name='Hidden_layer_2'))
## Second Layer
pool2 = tf. compat.v1.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
dnn_model.add(ks.layers.Dense(128, activation=hidden_activation_function, name='Hidden_layer_3'))

# dense layer
dnn_model.add(ks.layers.Dense(10, activation=output_activation_function, name='Output_layer'))
dnn_model.summary()

# train and evaluate the model
training_loss, training_accuracy = dnn_model.evaluate(training_images, training_labels)
print('Training Data Accuracy {}'.format(round(float(training_accuracy),2)))

# test the model
test_loss, test_accuracy = dnn_model.evaluate(test_images, test_labels)
print('Test Data Accuracy {}'.format(round(float(test_accuracy),2)))
