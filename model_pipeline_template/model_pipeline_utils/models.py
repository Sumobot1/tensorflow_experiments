import pdb
import tensorflow as tf
from model_pipeline_utils.estimator_hooks import SuperWackHook
from model_pipeline_utils.abstract_layers import input_layer, conv_2d_layer, max_pool_2d_layer, flatten_layer, dense_layer, dropout_layer, mean_softmax_cross_entropy_with_logits


def tf_model_outputs(logits, labels, predictions, mode, params):
    # If predict and not return_extimator, return predictions
    return tf.cond(mode, lambda: (mean_softmax_cross_entropy_with_logits(labels, logits, params), predictions), lambda: (tf.constant(0, dtype=tf.float32), predictions))


def cnn_model_fn(features, labels, mode, final_dropout_rate, params):
    """Model function for CNN."""
    # Try separable conv2d - Didn't work.
    # Don't wrap blocks in name scopes - the tensorboard graph looks wack...
    conv1 = conv_2d_layer(features, 77, [3, 3], "same", 'leaky_relu', 'he_normal', 'conv1', summary=params["histogram_summary"])
    # conv1 = tf.layers.conv2d(inputs=features, filters=77, kernel_size=[3, 3], padding="same", activation=tf.nn.leaky_relu, kernel_initializer=tf.keras.initializers.he_normal(), name="conv1")
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name="pool1")
    norm1 = tf.layers.batch_normalization(inputs=pool1, axis=3, name="norm1")
    if params["histogram_summary"]:
        # tf.summary.histogram("conv1", conv1)
        tf.summary.histogram("pool1", pool1)
        tf.summary.histogram("norm1", norm1)

    conv2 = tf.layers.conv2d(inputs=norm1, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.leaky_relu, kernel_initializer=tf.keras.initializers.he_normal(), name="conv2")
    norm2 = tf.layers.batch_normalization(inputs=conv2, axis=3, name="norm2")
    if params["histogram_summary"]:
        tf.summary.histogram("conv2", conv2)
        tf.summary.histogram("norm2", norm2)

    conv3 = tf.layers.conv2d(inputs=norm2, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.leaky_relu, kernel_initializer=tf.keras.initializers.he_normal(), name="conv3")
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2, name="pool3")
    norm3 = tf.layers.batch_normalization(inputs=pool3, axis=3, name="norm3")
    if params["histogram_summary"]:
        tf.summary.histogram("conv3", conv3)
        tf.summary.histogram("pool3", pool3)
        tf.summary.histogram("norm3", norm3)

    conv4 = tf.layers.conv2d(inputs=norm3, filters=32, kernel_size=[3, 3], padding="same", activation=tf.nn.leaky_relu, kernel_initializer=tf.keras.initializers.he_normal(), name="conv4")
    norm4 = tf.layers.batch_normalization(inputs=conv4, axis=3, name="norm4")
    if params["histogram_summary"]:
        tf.summary.histogram("conv4", conv4)
        tf.summary.histogram("norm4", norm4)

    conv5 = tf.layers.conv2d(inputs=norm4, filters=32, kernel_size=[3, 3], padding="same", activation=tf.nn.leaky_relu, kernel_initializer=tf.keras.initializers.he_normal(), name="conv5")
    pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2, name="pool5")
    norm5 = tf.layers.batch_normalization(inputs=pool5, axis=3, name="norm5")
    if params["histogram_summary"]:
        tf.summary.histogram("conv5", conv5)
        tf.summary.histogram("pool5", pool5)
        tf.summary.histogram("norm5", norm5)

    pool5_flat = tf.layers.flatten(inputs=norm5, name="pool5_flat")
    dense = tf.layers.dense(inputs=pool5_flat, units=42, activation=tf.nn.leaky_relu, name="dense")
    dropout = tf.layers.dropout(inputs=dense, rate=0.9, training=mode, name="dropout")
    if params["histogram_summary"]:
        tf.summary.histogram("pool5_flat", pool5_flat)
        tf.summary.histogram("dense", dense)
        tf.summary.histogram("dropout", dropout)

    logits = tf.layers.dense(inputs=dropout, units=2, name="logits")

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1, name="prediction"),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    return tf_model_outputs(logits, labels, predictions, mode, params)


def fast_cnn_model_fn(features, labels, mode, params):
    """Model function for CNN."""
    # Convolutional Layer #1
    conv1 = conv_2d_layer(features, 32, [3, 3], "conv1", "same", "leaky_relu")
    # Pooling Layer #1
    pool_1 = max_pool_2d_layer(conv1, [2, 2], 2)
    # Dense Layer
    pool1_flat = flatten_layer(pool_1)
    # Logits Layer
    logits = dense_layer(pool1_flat, 2)
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    return tf_model_outputs(logits, labels, predictions, mode, params)
