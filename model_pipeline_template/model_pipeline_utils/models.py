import tensorflow as tf
from model_pipeline_utils.estimator_hooks import SuperWackHook
from model_pipeline_utils.abstract_layers import input_layer, conv_2d, max_pool_2d, flatten, dense, dropout


def tf_model_estimator(logits, labels, predictions, mode, params):
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    # Note:
    # tf.losses.sparse_softmax_cross_entropy is depricated and will be removed soon
    # tf.nn.sparse_softmax_cross_entropy_with_logits works differently and returns a tensor instead of a differentiable value.
    # It needs to be wrapped in tf.reduce_mean to work properly
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits), name='loss_layer')
    if params["loss_summary"]:
        tf.summary.scalar("loss", loss)
    if not params["return_estimator"]:
        return (loss, predictions)
    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=tf.argmax(input=labels, axis=1), predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def cnn_model_fn(features, labels, mode, params):
    """Model function for CNN."""
    input_layer = tf.reshape(features, [-1, 80, 80, 3], name="input_layer")
    # Try separable conv2d - Didn't work.
    # Don't wrap blocks in name scopes - the tensorboard graph looks wack...
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=77, kernel_size=[3, 3], padding="same", activation=tf.nn.leaky_relu, kernel_initializer=tf.keras.initializers.he_normal(), name="conv1")
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name="pool1")
    norm1 = tf.layers.batch_normalization(inputs=pool1, axis=3, name="norm1")
    if params["histogram_summary"]:
        tf.summary.histogram("conv1", conv1)
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
    dropout = tf.layers.dropout(inputs=dense, rate=0.9, training=mode == tf.estimator.ModeKeys.TRAIN, name="dropout")
    if params["histogram_summary"]:
        tf.summary.histogram("pool5_flat", pool5_flat)
        tf.summary.histogram("dense", dense)
        tf.summary.histogram("dropout", dropout)

    logits = tf.layers.dense(inputs=dropout, units=2, name="logits")

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    return tf_model_estimator(logits, labels, predictions, mode, params)


def fast_cnn_model_fn(features, labels, mode, params):
    image_batch = tf.placeholder(shape=[None, 80, 80, 3], dtype='float32', name="image_batch")
    label_batch = tf.placeholder(shape=[None, 2], dtype='float32', name="label_batch")
    """Model function for CNN."""
    # Convolutional Layer #1
    conv1 = conv_2d(features, 32, [3, 3], "same", "leaky_relu")
    # Pooling Layer #1
    pool_1 = max_pool_2d(conv1, [2, 2], 2)
    # Dense Layer
    pool1_flat = flatten(pool_1)
    # Logits Layer
    logits = dense(pool1_flat, 2)
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    return tf_model_estimator(logits, labels, predictions, mode, params)
