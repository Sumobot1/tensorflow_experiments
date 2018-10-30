# Adapted from https://github.com/antonilo/TensBlur
import os
import pdb
import numpy as np
import tensorflow as tf
import scipy.stats as st


from PIL import Image

test_data_dir = '/home/michael/hard_drive/datasets/dogs_vs_cats_data/test'


def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis=2)
    return out_filter


def make_gauss_var(size, sigma, c_i):
    kernel = gauss_kernel(size, sigma, c_i)
    var = tf.convert_to_tensor(kernel)
    return var


def conv(input_ting, name, filter_size, sigma, padding='SAME'):
    # Get the number of channels in the input
    c_i = input_ting.get_shape().as_list()[3]
    # Convolution for a given input and kernel
    convolve = lambda i, k: tf.nn.depthwise_conv2d(i, k, [1, 1, 1, 1], padding=padding)
    with tf.variable_scope(name) as scope:
        kernel = make_gauss_var(filter_size, sigma, c_i)
        output = convolve(input_ting, kernel)
        return output


def smooth():
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        image = Image.open(os.path.join(test_data_dir, '1.jpg'))
        im = np.squeeze(np.uint8(image))
        image = np.array(image, dtype=np.float32)
        image = image.reshape((1, image.shape[0], image.shape[1], 3))
        image_tensor = tf.convert_to_tensor(image)
        smoothed_image = conv(image_tensor, "conv", 13, 2.0)
        smoothed = smoothed_image.eval()
        smoothed = smoothed / np.max(smoothed)
        out_image = np.squeeze(smoothed)
        out_image = Image.fromarray(np.squeeze(np.uint8(out_image * 255)))
        out_image.show()
        Image.fromarray(im).show()


def main(argv=None):
    smooth()


if __name__ == '__main__':
    tf.app.run()
