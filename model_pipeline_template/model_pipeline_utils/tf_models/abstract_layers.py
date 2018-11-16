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


def _layer_histogram(layer_name, layer, summary=False):
    if summary:
        tf.summary.histogram(layer_name, layer)


# input_layer: Layer to be inputted
# output_shape: Array representing shape to output (ex. [-1, 80, 80, 3])
def reshape_layer(input_layer, output_shape):
    return tf.reshape(input_layer, output_shape)


# filter_size: number - how many nodes in convolution (32 - 32x32 convolution)
# kernel_size: array - dimensions of kernel (ex. [3, 3])
def conv_2d_layer(input_layer, filter_size, kernel_size, padding_type, activation_fn, initializer, layer_name, summary=False):
    layer = tf.layers.conv2d(inputs=input_layer, filters=filter_size, kernel_size=kernel_size, padding=padding_type, activation=activation_fns[activation_fn], kernel_initializer=kernel_initializers[initializer], name=layer_name)
    _layer_histogram(layer_name, layer, summary)
    # if summary:
    #     tf.summary.histogram(layer_name, layer)
    return layer


def max_pool_2d_layer(input_layer, pool_size, stride_size, layer_name, summary=False):
    layer = tf.layers.max_pooling2d(inputs=input_layer, pool_size=pool_size, strides=stride_size)
    _layer_histogram(layer_name, layer, summary)
    # if summary:
    #     tf.summary.histogram(layer_name, layer)
    return layer


def batch_norm_layer(input_layer, axis_num, layer_name, summary=False):
    layer = tf.layers.batch_normalization(inputs=input_layer, axis=axis_num, name=layer_name)
    _layer_histogram(layer_name, layer, summary)
    # if summary:
    #     tf.summary.histogram(layer_name, layer)
    return layer


def flatten_layer(input_layer, layer_name, summary=False):
    layer = tf.layers.flatten(inputs=input_layer, name=layer_name)
    _layer_histogram(layer_name, layer, summary)
    return layer


def dropout_layer(input_layer, dropout_rate, training_mode, layer_name, summary=False):
    layer = tf.layers.dropout(inputs=input_layer, rate=dropout_rate, training=training_mode, name=layer_name)
    _layer_histogram(layer_name, layer, summary)
    # if summary:
    #     tf.summary.histogram(layer_name, layer)
    return layer


def dense_layer(input_layer, num_units, layer_name, activation_fn='none', summary=False):
    layer = tf.layers.dense(inputs=input_layer, units=num_units, activation=activation_fns[activation_fn])
    _layer_histogram(layer_name, layer, summary)
    return layer


def mean_softmax_cross_entropy_with_logits(labels, logits, params):
    # Note:
    # tf.losses.sparse_softmax_cross_entropy is depricated and will be removed soon
    # tf.nn.sparse_softmax_cross_entropy_with_logits works differently and returns a tensor instead of a differentiable value.
    # It needs to be wrapped in tf.reduce_mean to work properly
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits), name='loss_layer')
    if params["loss_summary"]:
        tf.summary.scalar("loss", loss)
    return loss


def softmax_classifier_output(logits, labels, predictions, mode, params):
    # If predict and not return_extimator, return predictions
    return tf.cond(mode,
                   lambda: (mean_softmax_cross_entropy_with_logits(labels, logits, params), predictions),
                   lambda: (tf.constant(0, dtype=tf.float32), predictions))