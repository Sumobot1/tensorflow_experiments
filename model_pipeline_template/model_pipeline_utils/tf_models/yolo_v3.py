import tensorflow as tf
import os
import io
from collections import defaultdict
import numpy as np
from model_pipeline_utils.tf_models.abstract_layers import conv_2d_layer, batch_norm_layer, leaky_relu_layer,\
    upsampling_2d_layer, concat_layer, l1_l2_reg, darknet_conv2d, padding_layer, add_layer


class YOLOV3:
    def __init__(self):
        self.layers = []

    def append_layer(self, layer):
        self.layers.append(layer)
        return layer

    def darknet_conv2d(self, input_layer, num_filters, kernel_size, layer_name,
                       use_bias=True, strides=(1, 1), summary=False):
        return self.append_layer(
            darknet_conv2d(input_layer, num_filters, kernel_size, layer_name, use_bias, strides, summary))

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
        darknet_conv = self.darknet_conv2d(input_layer, num_filters, kernel_size,
                                           "darknet_conv_{}".format(layer_num), False, strides)
        batch_norm = self.batch_norm_layer(darknet_conv, -1, "batch_norm_{}".format(layer_num), summary=False)
        return self.leaky_relu_layer(batch_norm, "leaky_relu_{}".format(layer_num), alpha=0.3, summary=False)

    def resblock_body(self, input_layer, num_filters, num_blocks, layer_num):
        zero_pad = self.padding_layer(input_layer, paddings=((0, 0), (1, 0), (1, 0), (0, 0)), mode="CONSTANT",
                                      layer_name="pad_{}".format(layer_num))
        print(zero_pad)
        # zero_pad = tf.keras.layers.ZeroPadding2D()
        leaky_darknet = self.darknet_conv2d_bn_leaky(zero_pad, num_filters, (3, 3), (2, 2),
                                                     "{}_start".format(layer_num))
        print(leaky_darknet)
        current_network = leaky_darknet
        for i in range(num_blocks):
            conv_block = self.darknet_conv2d_bn_leaky(current_network, num_filters // 2, (1, 1),
                                                      layer_num="{}_part_{}".format((layer_num + i), 1))
            conv_block = self.darknet_conv2d_bn_leaky(conv_block, num_filters, (3, 3),
                                                      layer_num="{}_part_{}".format((layer_num + i), 2))
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
        x = self.darknet_conv2d_bn_leaky(input_layer, num_filters, (1, 1),
                                         layer_num="x_{}_conv_output_layer_{}".format(output_identifier, 0))
        x = self.darknet_conv2d_bn_leaky(x, num_filters * 2, (3, 3),
                                         layer_num="x_{}_conv_output_layer_{}".format(output_identifier, 1))
        x = self.darknet_conv2d_bn_leaky(x, num_filters, (1, 1),
                                         layer_num="x_{}_conv_output_layer_{}".format(output_identifier, 2))
        x = self.darknet_conv2d_bn_leaky(x, num_filters * 2, (3, 3),
                                         layer_num="x_{}_conv_output_layer_{}".format(output_identifier, 3))
        x = self.darknet_conv2d_bn_leaky(x, num_filters, (1, 1),
                                         layer_num="x_{}_conv_output_layer_{}".format(output_identifier, 4))

        y = self.darknet_conv2d_bn_leaky(x, num_filters * 2, (3, 3),
                                         layer_num="y_{}_conv_output_layer_{}".format(output_identifier, 0))
        y = self.darknet_conv2d(y, num_outputs, (1, 1),
                                layer_name="y_{}_conv_output_layer_final_{}".format(output_identifier, 1))
        return x, y

    def yolo_body(self, image_input, num_anchors, num_classes, load_weights=False):
        """
        image_input: [None, None, 3]
        num_anchors: num_actual_anchors / 3 - YOLOv3 detects at 3 different scales.
                     The number of anchors are divided evenly into the 3 scales.
        num_classes
        """
        # 80 CLASSES, 9 ANCHORS
        # image_input = tf.placeholder(shape=(None, None, 3), dtype=tf.float32)
        darknet_output = self.darknet_body(image_input)
        x, y1 = self.make_last_layers(darknet_output, 512, num_anchors*(num_classes + 5), 1)
        x = self.darknet_conv2d_bn_leaky(x, 256, (1, 1), layer_num="x_1_conv_upsampling_0")
        x = self.upsampling_2d_layer(x, 2, "x_1_upsampling_1")
        x = self.concat_layer([x, self.layers[151]], -1, "x_1_concat")
        x, y2 = self.make_last_layers(x, 256, num_anchors * (num_classes + 5), 2)
        x = self.darknet_conv2d_bn_leaky(x, 128, (1, 1), layer_num="x_2_conv_upsampling_0")
        x = self.upsampling_2d_layer(x, 2, "x_2_upsampling_1")
        x = self.concat_layer([x, self.layers[91]], -1, "x_2_concat")
        x, y3 = self.make_last_layers(x, 128, num_anchors * (num_classes + 5), 3)
        return [y1, y2, y3]


def yolo_v3(image_input, num_anchors, num_classes, load_weights=False):
    # import tensorflow as tf; import model_pipeline_utils.tf_models.yolo_v3 as yolo3; image_input = tf.placeholder(shape=(1, 320, 320, 3), dtype=tf.float32)
    # Image size must be a multiple of 32!!!  Otherwise this thing won't work...
    # May need to send in a train_body flag or something.  Might want to freeze layers.
    detections = YOLOV3().yolo_body(image_input, num_anchors, num_classes, load_weights)
    if load_weights:
        # NEED TO ACTUALLY LOAD THESE WEIGHTS... ================================================
        yolo_weights = load_yolo_weights()
        print("here")
        import pdb; pdb.set_trace()
    # boxes = detections_boxes(detections)
    import pdb; pdb.set_trace()
    return

def yolo_loss(yolo_outputs, y_true, anchors, num_classes, ignore_thresh, print_loss):
    num_layers = len(anchors) // 3
    # Assume num layers = 3?
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    input_shape = tf.cast(tf.shape(yolo_outputs[0])[1:3]*32, tf.float32)
    grid_shapes = [tf.cast(tf.shape(yolo_outputs[l])[1:3], tf.float32) for l in range(num_layers)]
    loss = 0
    m = tf.shape(yolo_outputs[0])[0]
    mf = tf.cast(m, tf.float32)
    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]
        grid, raw_pred, pred_xy, pred_wh = output_tensors_to_bboxes(yolo_outputs[l], anchors[anchor_mask[l]],
                                                                    num_classes, input_shape, calc_loss=True)
        pass
#     TODO: Not finished...


def output_tensors_to_bboxes(feats, anchors, num_classes, input_shape, calc_loss=False):
    anchors_tensor = tf.reshape(tf.constant(anchors), [1, 1, 1, len(anchors), 2])
    # When converting to bbox coordinates: by = sigma(ty)+cy -> sigma(ty) is the offset of the center of the box from
    # the current cell -> cy is the offset of the cell from the top left corner of the box.
    # ^ going to need to get cx and cy to get this working...
    # grid_y = tf.tile()
# def detections_boxes(detections):
#     """
#     Converts center x, center y, width and height values to coordinates of top left and bottom right points.
#
#     :param detections: outputs of YOLO v3 detector of shape (?, 10647, (num_classes + 5))
#     :return: converted detections of same shape as input
#     """
#     import pdb; pdb.set_trace()
#     center_x, center_y, width, height, attrs = tf.split(detections, [1, 1, 1, 1, -1], axis=-1)
#     w2 = width / 2
#     h2 = height / 2
#     x0 = center_x - w2
#     y0 = center_y - h2
#     x1 = center_x + w2
#     y1 = center_y + h2
#
#     boxes = tf.concat([x0, y0, x1, y1], axis=-1)
#     detections = tf.concat([boxes, attrs], axis=-1)
#     return detections


def load_yolo_weights():
    weights_path = 'data/yolov3.weights'
    if not os.path.exists(weights_path):
        os.system('wget -O {} https://pjreddie.com/media/files/yolov3.weights'.format(weights_path))
    # Load weights and config.
    print('Loading weights.')
    weights_file = open(weights_path, 'rb')
    major, minor, revision, _, _ = np.fromfile(weights_file, dtype=np.int32, count=5)
    weights = np.fromfile(weights_file, dtype=np.float32)
    var_list = tf.global_variables()
    ptr = 0
    i = 0
    assign_ops = []
    while i < len(var_list) - 1:
        var1 = var_list[i]
        var2 = var_list[i + 1]
        # import pdb; pdb.set_trace()
        # do something only if we process conv layer
        if 'conv' in var1.name.split('/')[-2]:
            # check type of next layer
            if 'batch_norm' in var2.name.split('/')[-2]:
                # load batch norm params
                gamma, beta, mean, var = var_list[i + 1:i + 5]
                batch_norm_vars = [beta, gamma, mean, var]
                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(tf.assign(var, var_weights, validate_shape=True))

                # we move the pointer by 4, because we loaded 4 variables
                i += 4
            elif 'conv' in var2.name.split('/')[-2]:
                # load biases
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
                if "final" in var2.name.split('/')[-2]:
                    # Regardless of the shape of the output, coco weights assume bias_params should be 255 here
                    ptr += 255
                else:
                    ptr += bias_params
                assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))

                # we loaded 1 variable
                i += 1
            else:
                print("else")
                import pdb; pdb.set_trace()
            # we can load weights of conv layer
            shape = var1.shape.as_list()
            num_params = np.prod(shape)

            var_weights = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
            # remember to transpose to column-major
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            if "final" in var2.name.split('/')[-2]:
                # Regardless of the shape fo the output, coco weights assume num params should have 255 as last index
                ptr += np.prod(var1.shape.as_list()[:-1] + [255,])
            else:
                ptr += num_params
            assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
            i += 1
    return assign_ops
