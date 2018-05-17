from random import shuffle
import glob
import pdb
import cv2
import numpy as np
import sys
import tensorflow as tf
from tqdm import tqdm
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras import models
from tensorflow.python.keras import layers
from tensorflow.python.keras.preprocessing import image
from tensorflow.python import keras
from PIL import Image
import os
import shutil


IMAGE_WIDTH = 80
IMAGE_HEIGHT = 80

def load_image(addr):
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    img = cv2.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img


# Pulled from Tensorflow's TFRecord documentation
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# Pulled from Tensorflow's TFRecord documentation
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# Modified from: https://www.dlology.com/blog/how-to-leverage-tensorflows-tfrecord-to-train-keras-model/
def write_tfrecord_file(addrs, labels, data_type, max_records=sys.maxsize):
    filename = '{}.tfrecords'.format(data_type)
    max_records = min(max_records, len(addrs))
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(filename)
    for i in tqdm(range(max_records)):
        img = load_image(addrs[i])
        label = labels[i]
        feature = {'{}/label'.format(data_type): _int64_feature(label), '{}/image'.format(data_type): _bytes_feature(tf.compat.as_bytes(img.tostring()))}
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
    writer.close()
    sys.stdout.flush()


# Modified from: https://www.dlology.com/blog/how-to-leverage-tensorflows-tfrecord-to-train-keras-model/
def generate_tfrecords(cat_dog_train_path):
    # read addresses and labels from the 'train' folder
    addrs = glob.glob(cat_dog_train_path)
    labels = [0 if 'cat' in addr.split('/')[-1] else 1 for addr in addrs]  # 0 = Cat, 1 = Dog
    # We will shuffle the dataset on read
    # Python zip() - Takes in n iterables and returns a list of tuples.
    # Each tuple is created from the ith element from each iterable

    # Ex:
    # list_a = [1, 2, 3, 4, 5]
    # list_b = ['a', 'b', 'c', 'd', 'e']
    # zipped_list = zip(list_a, list_b)
    # print zipped_list # [(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd'), (5, 'e')]

    # If the iterables are different lengths, tuple list length will be equal
    # to the length of the shortest list

    # Divide the hata into 60% train, 20% validation, and 20% test
    train_addrs = addrs[0:int(0.6 * len(addrs))]
    train_labels = labels[0:int(0.6 * len(labels))]
    val_addrs = addrs[int(0.6 * len(addrs)):int(0.8 * len(addrs))]
    val_labels = labels[int(0.6 * len(addrs)):int(0.8 * len(addrs))]
    test_addrs = addrs[int(0.8 * len(addrs)):]
    test_labels = labels[int(0.8 * len(labels)):]

    # train_filename = 'train.tfrecords'  # address to save the TFRecords file
    write_tfrecord_file(train_addrs, train_labels, 'train', 1000)
    write_tfrecord_file(val_addrs, val_labels, 'val', 1000)
    write_tfrecord_file(test_addrs, test_labels, 'test', 1000)


# Modified from: https://www.dlology.com/blog/how-to-leverage-tensorflows-tfrecord-to-train-keras-model/
def imgs_input_fn(filenames, data_type, perform_shuffle=False, repeat_count=1, batch_size=1):
    def _parse_function(serialized):
        features = {'{}/label'.format(data_type): tf.FixedLenFeature([], tf.int64),
                    '{}/image'.format(data_type): tf.FixedLenFeature([], tf.string)}
        # Parse the serialized data so we get a dict with our data.
        parsed_example = tf.parse_single_example(serialized=serialized, features=features)
        # Get the image as raw bytes.
        image_raw = parsed_example['{}/image'.format(data_type)]
        label = tf.cast(parsed_example['{}/label'.format(data_type)], tf.int32)
        # Decode the raw bytes so it becomes a tensor with type.
        image = tf.decode_raw(image_raw, tf.float32)
        image = tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
        # Don't know if we need to center the image in this case...
        # image = tf.subtract(image, 116.779) # Zero-center by mean pixel
        return image, label
    
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.
    dataset = dataset.map(_parse_function)
    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times
    dataset = dataset.batch(batch_size)  # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(inputs=features, filters=32, kernel_size=[3, 3], padding="valid", activation=tf.nn.relu)
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3], padding="valid", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    # Dense Layer
    pool2_flat = tf.layers.flatten(inputs=pool2)
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
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

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Note:
    # tf.losses.sparse_softmax_cross_entropy is depricated and will be removed soon
    # tf.nn.sparse_softmax_cross_entropy_with_logits works differently and returns a tensor instead of a differentiable value.
    # It needs to be wrapped in tf.reduce_mean to work properly

    # loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits), name='ting')

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(argv):
    # Need to set logging verbosity to INFO level or training loss will not print
    tf.logging.set_verbosity(tf.logging.INFO)
    # Training data needs to be split into training, validation, and testing sets
    # This needs to be a complete (not relative) path, or glob will run into issues

    cat_dog_train_path = '/home/michael/Documents/DataSets/dogs_vs_cats_data/*.jpg' if '--laptop' in argv else '/home/michael/hard_drive/datasets/dogs_vs_cats_data/train/*.jpg'
    generate_tfrecords(cat_dog_train_path)

    next_example, next_label = imgs_input_fn(['train.tfrecords'], 'train', perform_shuffle=True, repeat_count=5, batch_size=20)
    
    # A good way to debug programs like this is to run a tf.InteractiveSession()
    # sess = tf.InteractiveSession()

    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model5")
    tensors_to_log = {"probabilities": "ting"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    training_batch_size = 1 if '--laptop' in argv else 20
    mnist_classifier.train(input_fn=lambda: imgs_input_fn(['train.tfrecords'], 'train', perform_shuffle=True, repeat_count=500, batch_size=training_batch_size), steps=50000, hooks=[logging_hook])


if __name__ == "__main__":
    main(sys.argv)












