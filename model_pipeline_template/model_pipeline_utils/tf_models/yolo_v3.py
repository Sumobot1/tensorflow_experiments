import tensorflow as tf
from model_pipeline_utils.tf_models.abstract_layers import conv_2d_layer, batch_norm_layer, leaky_relu_layer,\
    upsampling_2d_layer, concat_layer, l1_l2_reg, darknet_conv2d, padding_layer, add_layer


class YOLOV3:
    def __init__(self):
        self.layers = []

    def append_layer(self, layer):
        self.layers.append(layer)
        return layer

    def darknet_conv2d(self, input_layer, num_filters, kernel_size, layer_name, use_bias=True, strides=(1, 1), summary=False):
        return self.append_layer(darknet_conv2d(input_layer, num_filters, kernel_size, layer_name, use_bias, strides, summary))

    def batch_norm_layer(self, input_layer, axis_num, layer_name, summary=False):
        return self.append_layer(batch_norm_layer(input_layer, axis_num, layer_name, summary=summary))

    def leaky_relu_layer(self, input_layer, layer_name, alpha=0.3, summary=False):
        return self.append_layer(leaky_relu_layer(input_layer, layer_name, alpha=alpha, summary=summary))

    def padding_layer(self, input_layer, paddings, mode, layer_name, summary=False):
        return self.append_layer(padding_layer(input_layer, paddings, mode, layer_name, summary=summary))

    def upsampling_2d_layer(self, input_layer, multiplier, layer_name, summary=False):
        return self.append_layer(upsampling_2d_layer(input_layer, multiplier, layer_name, summary))

    def add_layer(self, layers):
        return self.append_layer(add_layer(layers))

    def concat_layer(self, values, axis, layer_name, summary=False):
        return self.append_layer(concat_layer(values, axis, layer_name, summary))

    def darknet_conv2d_bn_leaky(self, input_layer, num_filters, kernel_size, strides=(1, 1), layer_num=None):
        darknet_conv = self.darknet_conv2d(input_layer, num_filters, kernel_size, "darknet_conv_{}".format(layer_num), False, strides)
        batch_norm = self.batch_norm_layer(darknet_conv, -1, "batch_norm_{}".format(layer_num), summary=False)
        return self.leaky_relu_layer(batch_norm, "leaky_relu_{}".format(layer_num), alpha=0.3, summary=False)

    def resblock_body(self, input_layer, num_filters, num_blocks, layer_num):
        zero_pad = self.padding_layer(input_layer, paddings=((0, 0), (1, 0), (1, 0), (0, 0)), mode="CONSTANT", layer_name="pad_{}".format(layer_num))
        print(zero_pad)
        # zero_pad = tf.keras.layers.ZeroPadding2D()
        leaky_darknet = self.darknet_conv2d_bn_leaky(zero_pad, num_filters, (3, 3), (2, 2), "{}_start".format(layer_num))
        print(leaky_darknet)
        current_network = leaky_darknet
        for i in range(num_blocks):
            conv_block = self.darknet_conv2d_bn_leaky(current_network, num_filters // 2, (1, 1), layer_num="{}_part_{}".format((layer_num + i), 1))
            conv_block = self.darknet_conv2d_bn_leaky(conv_block, num_filters, (3, 3), layer_num="{}_part_{}".format((layer_num + i), 2))
            current_network = self.add_layer([current_network, conv_block])
            print(current_network)
        return current_network

    def darknet_body(self, image_input):
        x = self.darknet_conv2d_bn_leaky(image_input, 32, (3, 3), layer_num=0)
        x = self.resblock_body(x, 64, 1, layer_num=1)
        x = self.resblock_body(x, 128, 2, layer_num=2)
        x = self.resblock_body(x, 256, 8, layer_num=4)
        x = self.resblock_body(x, 512, 8, layer_num=12)
        x = self.resblock_body(x, 1024, 4, layer_num=20)
        return x

    def make_last_layers(self, input_layer, num_filters, num_outputs, output_identifier):
        x = self.darknet_conv2d_bn_leaky(input_layer, num_filters, (1, 1), layer_num="x_{}_output_layer_{}".format(output_identifier, 0))
        x = self.darknet_conv2d_bn_leaky(x, num_filters * 2, (3, 3), layer_num="x_{}_output_layer_{}".format(output_identifier, 1))
        x = self.darknet_conv2d_bn_leaky(x, num_filters, (1, 1), layer_num="x_{}_output_layer_{}".format(output_identifier, 2))
        x = self.darknet_conv2d_bn_leaky(x, num_filters * 2, (3, 3), layer_num="x_{}_output_layer_{}".format(output_identifier, 3))
        x = self.darknet_conv2d_bn_leaky(x, num_filters, (1, 1), layer_num="x_{}_output_layer_{}".format(output_identifier, 4))

        y = self.darknet_conv2d_bn_leaky(x, num_filters * 2, (3, 3), layer_num="y_{}_output_layer_{}".format(output_identifier, 0))
        y = self.darknet_conv2d(y, num_outputs, (1, 1), layer_name="y_{}_output_layer_{}".format(output_identifier, 1))
        return x, y

    def yolo_body(self, image_input, num_anchors, num_classes):
        """
        image_input: [None, None, 3]
        num_anchors: num_actual_anchors / 3 - YOLOv3 detects at 3 different scales.
                     The number of anchors are divided evenly into the 3 scales.
        num_classes
        """
        # image_input = tf.placeholder(shape=(None, None, 3), dtype=tf.float32)
        darknet_output = self.darknet_body(image_input)
        x, y1 = self.make_last_layers(darknet_output, 512, num_anchors*(num_classes + 5), 1)
        x = self.darknet_conv2d_bn_leaky(x, 256, (1, 1), layer_num="x_1_upsampling_0")
        x = self.upsampling_2d_layer(x, 2, "x_1_upsampling_1")
        print("Before first concat")
        import pdb; pdb.set_trace()
        x = self.concat_layer([x, self.layers[151]], -1, "x_1_concat")
        import pdb; pdb.set_trace()
        x, y2 = self.make_last_layers(x, 256, num_anchors * (num_classes + 5), 2)
        x = self.darknet_conv2d_bn_leaky(x, 128, (1, 1), layer_num="x_2_upsampling_0")
        x = self.upsampling_2d_layer(x, 2, "x_2_upsampling_1")
        print("Before second concat")
        import pdb; pdb.set_trace()
        x = self.concat_layer([x, self.layers[91]], -1, "x_2_concat")
        x, y3 = self.make_last_layers(x, 128, num_anchors * (num_classes + 5), 3)
        import pdb;
        pdb.set_trace()
        return [y1, y2, y3]
#         Need to concat layers now...


def yolo_v3(image_input, num_anchors, num_classes):
    # import tensorflow as tf; import model_pipeline_utils.tf_models.yolo_v3 as yolo3; image_input = tf.placeholder(shape=(1, 300, 300, 3), dtype=tf.float32)
    # Image size must be a multiple of 32!!!  Otherwise this thing won't work...
    return YOLOV3().yolo_body(image_input, num_anchors, num_classes)
