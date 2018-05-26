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
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.python import keras
from PIL import Image
import os
import shutil
import multiprocessing as mp
from random import randint
import imutils
from PIL import Image, ImageFilter


IMAGE_WIDTH = 80
IMAGE_HEIGHT = 80
NUM_CPU_CORES = 6
AUGMENT = 5

# If there are too many zeros in the image, relu outputs zero, and will never recover.  Using Keras preprocessing function instead
# https://datascience.stackexchange.com/questions/21955/tensorflow-regression-model-giving-same-prediction-every-time
# https://github.com/keras-team/keras/issues/3687
# https://github.com/hellochick/PSPNet-tensorflow/issues/7
# https://datascience.stackexchange.com/questions/5706/what-is-the-dying-relu-problem-in-neural-networks
def load_image(addr, augment_data):
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        rescale=1. / 255)
    images = []
    img = Image.open(addr).filter(ImageFilter.GaussianBlur(1)).resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    array = img_to_array(img)
    array_normalized = ((array - array.min()) * (1. / (array.max() - array.min()) * 1)).astype('float32')
    images.append(array_normalized)
    if augment_data:
        array = array.reshape((1,) + array.shape)
        i = 0
        for batch in datagen.flow(array, batch_size=1):
            if i >= AUGMENT - 1:
                break
            images.append(np.reshape(batch, [80, 80, 3]))
            i += 1
    return images


# Pulled from Tensorflow's TFRecord documentation
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# Pulled from Tensorflow's TFRecord documentation
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def parallel_write_tfrecord_file(addrs, labels, data_type, max_records=sys.maxsize, augment_data=False):
    print("Parallel write tfrecord file {}".format(data_type))
    num_images = min(max_records, len(addrs))
    processes = [mp.Process(target=write_tfrecord_file, args=(x, int(num_images / NUM_CPU_CORES * x), int(num_images / NUM_CPU_CORES * (x + 1)), addrs, labels, data_type)) for x in range(NUM_CPU_CORES)]
    for p in processes:
        p.start()
    # Exit the completed processes
    for p in processes:
        p.join()
    # write_tfrecord_file(0, int(num_images / NUM_CPU_CORES * 0), int(num_images / NUM_CPU_CORES * (0 + 1)), addrs, labels, data_type)


# Modified from: https://www.dlology.com/blog/how-to-leverage-tensorflows-tfrecord-to-train-keras-model/
def write_tfrecord_file(thread_num, start_index, end_index, addrs, labels, data_type):
    filename = '{}_{}.tfrecords'.format(data_type, thread_num)
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(filename)
    total_files = end_index - start_index
    # TODO: Need to provide some sort of indicator of progress (% complete)
    for i in range(start_index, end_index):
        if (i % 100 == 0):
            print('Thread {} completed {}/{}'.format(thread_num, i - start_index, total_files))
        augment_data = True if data_type == 'train' else False
        imgs = load_image(addrs[i], augment_data)
        for img in imgs:
            label = labels[i]
            feature = {'{}/label'.format(data_type): _bytes_feature(tf.compat.as_bytes(np.asarray(label).tostring())), '{}/image'.format(data_type): _bytes_feature(tf.compat.as_bytes(img.tostring()))}
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
    labels = [[0, 1] if 'cat' in addr.split('/')[-1] else [1, 0] for addr in addrs]  # 0 = Cat, 1 = Dog
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

    parallel_write_tfrecord_file(train_addrs, train_labels, 'train', max_records=10000)
    parallel_write_tfrecord_file(val_addrs, val_labels, 'val', max_records=10000)
    parallel_write_tfrecord_file(test_addrs, test_labels, 'test', max_records=10000)


# Modified from: https://www.dlology.com/blog/how-to-leverage-tensorflows-tfrecord-to-train-keras-model/
def imgs_input_fn(filenames, data_type, perform_shuffle=False, repeat_count=1, batch_size=1):
    def _parse_function(serialized):
        features = {'{}/label'.format(data_type): tf.FixedLenFeature([], tf.string),
                    '{}/image'.format(data_type): tf.FixedLenFeature([], tf.string)}
        # Parse the serialized data so we get a dict with our data.
        parsed_example = tf.parse_single_example(serialized=serialized, features=features)
        # Get the image as raw bytes.
        image_raw = parsed_example['{}/image'.format(data_type)]
        label = tf.decode_raw(parsed_example['{}/label'.format(data_type)], tf.int64)
        label = tf.cast(label, tf.float32)
        label = tf.reshape(label, [2])
        # Decode the raw bytes so it becomes a tensor with type.
        img = tf.reshape(tf.decode_raw(image_raw, tf.float32), [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
        # Don't know if we need to center the image in this case...
        # image = tf.subtract(image, 116.779) # Zero-center by mean pixel
        return img, label

    dataset = tf.data.TFRecordDataset(filenames=filenames)
    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.
    dataset = dataset.map(map_func=_parse_function, num_parallel_calls=NUM_CPU_CORES)
    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times
    dataset = dataset.batch(batch_size)  # Batch size to use
    # How many elements (in this case batches) get consumed per epoch?
    dataset = dataset.prefetch(buffer_size=2)
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels
