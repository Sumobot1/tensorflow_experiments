import tensorflow as tf
from tensorflow.python import keras

activation_fns = {
    'relu': tf.nn.relu,
    'leaky_relu': tf.nn.leaky_relu,
    'none': None
}
kernel_initializers = {
    'he_normal': tf.keras.initializers.he_normal()
}


# input_layer: Layer to be inputted
# output_shape: Array representing shape to output (ex. [-1, 80, 80, 3])
def input_layer(input_layer, output_shape):
    return tf.reshape(input_layer, output_shape)


# filter_size: number - how many nodes in convolution (32 - 32x32 convolution)
# kernel_size: array - dimensions of kernel (ex. [3, 3])
def conv_2d_layer(input_layer, filter_size, kernel_size, padding_type, activation_fn, initializer, layer_name, summary=False):
    conv_layer = tf.layers.conv2d(inputs=input_layer, filters=filter_size, kernel_size=kernel_size, padding=padding_type, activation=activation_fns[activation_fn], kernel_initializer=kernel_initializers[initializer], name=layer_name)
    if summary:
        tf.summary.histogram(layer_name, conv_layer)
    return conv_layer


def max_pool_2d_layer(input_layer, pool_size=[2, 2], stride_size=2):
    return tf.layers.max_pooling2d(inputs=input_layer, pool_size=pool_size, strides=stride_size)


def flatten_layer(input_layer):
    return tf.layers.flatten(inputs=input_layer)


def dropout_layer(input_layer, dropout_rate, training_mode):
    return tf.layers.dropout(inputs=input_layer, rate=dropout_rate, training=training_mode)


def dense_layer(input_layer, num_units, activation_fn='none'):
    return tf.layers.dense(inputs=input_layer, units=num_units, activation=activation_fns[activation_fn])
