import tensorflow as tf
from estimator_hooks import SuperWackHook
from abstract_layers import input_layer, conv_2d, max_pool_2d, flatten, dense, dropout

def tf_model_estimator(logits, labels, predictions, mode, params):
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Note:
    # tf.losses.sparse_softmax_cross_entropy is depricated and will be removed soon
    # tf.nn.sparse_softmax_cross_entropy_with_logits works differently and returns a tensor instead of a differentiable value.
    # It needs to be wrapped in tf.reduce_mean to work properly

    # loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    # ting = tf.reduce_mean(loss, name='loss_layer')
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits), name='loss_layer')
    acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(input=labels, axis=1), predictions=predictions["classes"], name='accuracy_layer')
    batch_size = tf.shape(logits)[0]
    tensors_to_log = {'loss': loss, 'accuracy': acc_op, 'batch_size': batch_size, 'logits[0]': logits[0][0], 'logits[1]': logits[0][1], 'labels[0]': labels[0][0], 'labels[1]': labels[0][1]}
    wack_hook = SuperWackHook(tensors_to_log, every_n_iter=50, total_num_steps=params['total_num_steps'])
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()#learning_rate=0.0001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks=[wack_hook])

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=tf.argmax(input=labels, axis=1), predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def cnn_model_fn(features, labels, mode, params):
    """Model function for CNN."""
    input_layer = tf.reshape(features, [-1, 80, 80, 3])
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[3, 3], padding="same", activation=tf.nn.leaky_relu)
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.leaky_relu, kernel_initializer=tf.keras.initializers.he_normal())
    conv2 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.leaky_relu, kernel_initializer=tf.keras.initializers.he_normal())
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    conv3 = tf.layers.conv2d(inputs=pool2, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.leaky_relu, kernel_initializer=tf.keras.initializers.he_normal())
    conv3 = tf.layers.conv2d(inputs=conv3, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.leaky_relu, kernel_initializer=tf.keras.initializers.he_normal())
    # Pooling Layer #1
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
    # Dense Layer
    pool2_flat = tf.layers.flatten(inputs=pool3)
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.leaky_relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=2)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    return tf_model_estimator(logits, labels, predictions, mode, params)


def fast_cnn_model_fn(features, labels, mode, params):
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
