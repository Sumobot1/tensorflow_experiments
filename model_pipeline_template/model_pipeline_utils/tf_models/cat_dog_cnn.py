import tensorflow as tf
from model_pipeline_utils.tf_models.abstract_layers import conv_2d_layer, max_pool_2d_layer, flatten_layer, dense_layer, dropout_layer, mean_softmax_cross_entropy_with_logits, batch_norm_layer, softmax_classifier_output


def cnn_model_fn(features, labels, mode, final_dropout_rate, params):
    """Model function for CNN."""
    # Try separable conv2d - Didn't work.
    # Don't wrap blocks in name scopes - the tensorboard graph looks wack...
    conv1 = conv_2d_layer(features, 77, [3, 3], "same", 'leaky_relu', 'he_normal', 'conv1',
                          summary=params["histogram_summary"])
    pool1 = max_pool_2d_layer(conv1, (2, 2), 2, 'pool1', params['histogram_summary'])
    norm1 = batch_norm_layer(pool1, 3, 'norm1', params['histogram_summary'])
    conv2 = conv_2d_layer(norm1, 64, [3, 3], "same", 'leaky_relu', 'he_normal', 'conv2',
                          summary=params["histogram_summary"])
    norm2 = batch_norm_layer(conv2, 3, 'norm2', params['histogram_summary'])

    conv3 = conv_2d_layer(norm2, 64, [3, 3], "same", 'leaky_relu', 'he_normal', 'conv3',
                          summary=params["histogram_summary"])
    pool3 = max_pool_2d_layer(conv3, (2, 2), 2, 'pool3', params['histogram_summary'])
    norm3 = batch_norm_layer(pool3, 3, 'norm3', params['histogram_summary'])

    conv4 = conv_2d_layer(norm3, 32, [3, 3], "same", 'leaky_relu', 'he_normal', 'conv4',
                          summary=params["histogram_summary"])
    norm4 = batch_norm_layer(conv4, 3, 'norm4', params['histogram_summary'])

    conv5 = conv_2d_layer(norm4, 32, [3, 3], "same", 'leaky_relu', 'he_normal', 'conv5',
                          summary=params["histogram_summary"])
    pool5 = max_pool_2d_layer(conv5, (2, 2), 2, 'pool5', params['histogram_summary'])
    norm5 = batch_norm_layer(pool5, 3, 'norm5', params['histogram_summary'])

    pool5_flat = flatten_layer(norm5, 'pool5_flat', params['histogram_summary'])
    dense = dense_layer(pool5_flat, 42, 'dense', 'leaky_relu', summary=params['histogram_summary'])
    dropout = dropout_layer(dense, final_dropout_rate, mode, 'dropout', params['histogram_summary'])

    logits = dense_layer(dropout, 2, 'logits', summary=params['histogram_summary'])

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1, name="prediction"),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    return softmax_classifier_output(logits, labels, predictions, mode, params)