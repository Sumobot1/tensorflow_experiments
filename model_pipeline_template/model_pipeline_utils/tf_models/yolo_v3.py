import tensorflow as tf
from model_pipeline_utils.tf_models.abstract_layers import conv_2d_layer, batch_norm_layer, leaky_relu_layer


def l1_l2_reg(l1=0.0, l2=0.0):
    return tf.keras.regularizers.L1L2(l1, l2)


def darknet_conv2d(input_layer, num_filters, kernel_size, layer_name, use_bias, strides=(1, 1), summary=False):
    padding = 'valid' if strides == (2, 2) else 'same'
    return conv_2d_layer(input_layer, num_filters, kernel_size, padding, None, 'glorot_uniform', layer_name,
                         strides=strides, kernel_regularizer=l1_l2_reg(l2=5e-4), use_bias=use_bias, summary=summary)


def darknet_conv2d_bn_leaky(input_layer, num_filters, kernel_size, strides=(1, 1), layer_num=None):
    # if layer_num is not int(layer_num) or layer_num < 0:
    #     raise("Layer num must be an int, and greater than 0")
    darknet_conv = darknet_conv2d(input_layer, num_filters, kernel_size, "darknet_conv_{}".format(layer_num), False, strides)
    batch_norm = batch_norm_layer(darknet_conv, -1, "batch_norm_{}".format(layer_num), summary=False)
    return leaky_relu_layer(batch_norm, "leaky_relu_{}".format(layer_num), alpha=0.3, summary=False)


def resblock_body(input_layer, num_filters, num_blocks, layer_num):
    zero_pad = tf.pad(input_layer, paddings=((0, 0), (1, 0), (1, 0), (0, 0)), mode="CONSTANT")
    print(zero_pad)
    # zero_pad = tf.keras.layers.ZeroPadding2D()
    leaky_darknet = darknet_conv2d_bn_leaky(zero_pad, num_filters, (3, 3), (2, 2), "{}_start".format(layer_num))
    print(leaky_darknet)
    current_network = leaky_darknet
    for i in range(num_blocks):
        conv_block = darknet_conv2d_bn_leaky(current_network, num_filters // 2, (1, 1), layer_num="{}_part_{}".format((layer_num + i), 1))
        conv_block = darknet_conv2d_bn_leaky(conv_block, num_filters, (3, 3), layer_num="{}_part_{}".format((layer_num + i), 2))
        current_network = tf.keras.layers.add([current_network, conv_block])
        print(current_network)
    return current_network


def darknet_body(image_input):
    x = darknet_conv2d_bn_leaky(image_input, 32, (3, 3), layer_num=0)
    x = resblock_body(x, 64, 1, layer_num=1)
    x = resblock_body(x, 128, 2, layer_num=2)
    x = resblock_body(x, 256, 8, layer_num=4)
    x = resblock_body(x, 512, 8, layer_num=12)
    x = resblock_body(x, 1024, 4, layer_num=20)
    return x


def yolo_body(image_input, num_anchors, num_classes):
    """
    image_input: [None, None, 3]
    num_anchors: num_actual_anchors / 3 - YOLOv3 detects at 3 different scales.
                 The number of anchors are divided evenly into the 3 scales.
    num_classes
    """
    # image_input = tf.placeholder(shape=(None, None, 3), dtype=tf.float32)
