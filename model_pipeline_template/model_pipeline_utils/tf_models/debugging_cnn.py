import tensorflow as tf
from model_pipeline_utils.tf_models.abstract_layers import conv_2d_layer, max_pool_2d_layer, flatten_layer, dense_layer, dropout_layer, mean_softmax_cross_entropy_with_logits, batch_norm_layer, softmax_classifier_output


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

    return softmax_classifier_output(logits, labels, predictions, mode, params)