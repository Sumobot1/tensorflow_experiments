import tensorflow as tf
from tensorflow.python import keras

activation_fns = {
    'relu': tf.nn.relu,
    'leaky_relu': tf.nn.leaky_relu,
    'none': None
}
kernel_initializers = {
    'he_normal': tf.keras.initializers.he_normal(),
    'glorot_uniform': tf.keras.initializers.glorot_uniform()
}


def _layer_histogram(layer_name, layer, summary=False):
    if summary:
        tf.summary.histogram(layer_name, layer)


# input_layer: Layer to be inputted
# output_shape: Array representing shape to output (ex. [-1, 80, 80, 3])
def reshape_layer(input_layer, output_shape):
    return tf.reshape(input_layer, output_shape)


# num_filters: number - how many nodes in convolution (32 - 32x32 convolution)
# kernel_size: array - dimensions of kernel (ex. [3, 3])
def conv_2d_layer(input_layer, num_filters, kernel_size, padding_type, activation_fn, initializer, layer_name,
                  strides=(1, 1), kernel_regularizer=None, use_bias=True, trainable=True, summary=False):
    layer = tf.layers.conv2d(inputs=input_layer,
                             filters=num_filters,
                             kernel_size=kernel_size,
                             strides=strides,
                             padding=padding_type,
                             activation=activation_fns[str(activation_fn).lower()],
                             use_bias=use_bias,
                             kernel_initializer=kernel_initializers[initializer],
                             kernel_regularizer=kernel_regularizer,
                             name=layer_name)
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


def leaky_relu_layer(input_layer, layer_name, alpha=0.3, summary=False):
    layer = tf.nn.leaky_relu(input_layer, alpha=alpha)
    _layer_histogram(layer_name, layer, summary)
    return layer


def upsampling_2d_layer(input_layer, multiplier, layer_name, summary=False):
    input_dims = input_layer.shape.as_list()
    layer = tf.image.resize_nearest_neighbor(input_layer, (multiplier*input_dims[1], multiplier*input_dims[2]))
    _layer_histogram(layer_name, layer, summary)
    return layer


def concat_layer(values, axis, layer_name, summary=False):
    layer = tf.concat(values, axis, name=layer_name)
    _layer_histogram(layer_name, layer, summary)
    return layer

def l1_l2_reg(l1=0.0, l2=0.0):
        return tf.keras.regularizers.L1L2(l1, l2)

    # ting = {"input_layer": params[0], "num_filters": params[1], "kernel_size": params[2], "layer_name": "ya", "use_bias": params[4], "strides": params[5]}
    # self.layers.append(layer(**ting))

def darknet_conv2d(input_layer, num_filters, kernel_size, layer_name, use_bias=True, strides=(1, 1), summary=False):
    padding = 'valid' if strides == (2, 2) else 'same'
    return conv_2d_layer(input_layer, num_filters, kernel_size, padding, None, 'glorot_uniform', layer_name,
                         strides=strides, kernel_regularizer=l1_l2_reg(l2=5e-4), use_bias=use_bias, summary=summary)

def padding_layer(input_layer, paddings, mode, layer_name, summary=False):
    layer = tf.pad(input_layer, paddings=paddings, mode=mode, name=layer_name)
    _layer_histogram(layer_name, layer, summary)
    return layer

def add_layer(layers):
    return tf.keras.layers.add(layers)