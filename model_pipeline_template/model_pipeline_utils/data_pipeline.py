from termcolor import cprint
from random import shuffle
import json
import glob
import numpy as np
import sys
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, img_to_array
import os
import shutil
import multiprocessing as mp
import time
from PIL import Image, ImageFilter


IMAGE_WIDTH = 80
IMAGE_HEIGHT = 80
NUM_CPU_CORES = 6
AUGMENT = 5


def load_image(addr, augment_data, image_dims):
    # If there are too many zeros in the image, relu will output zero and never recover.
    # Using Keras preprocessing function instead
    # https://datascience.stackexchange.com/questions/21955/tensorflow-regression-model-giving-same-prediction-every-time
    # https://github.com/keras-team/keras/issues/3687
    # https://github.com/hellochick/PSPNet-tensorflow/issues/7
    # https://datascience.stackexchange.com/questions/5706/what-is-the-dying-relu-problem-in-neural-networks
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
    # Really, we should be doing data augmentation on read for better randomness, but I like using my cpu for other
    # things while training (GPU training makes my computer lag a bit as is)
    img = Image.open(addr).filter(ImageFilter.GaussianBlur(1)).resize((image_dims[1], image_dims[0]))
    array = img_to_array(img)
    array_normalized = ((array - array.min()) * (1. / 255.0 * 1)).astype('float32')
    images.append(array_normalized)
    if augment_data:
        array = array.reshape((1,) + array.shape)
        i = 0
        for batch in datagen.flow(array, batch_size=1):
            if i >= AUGMENT - 1:
                break
            images.append(np.reshape(batch, image_dims))
            i += 1
    return images


# Pulled from Tensorflow's TFRecord documentation
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# Pulled from Tensorflow's TFRecord documentation
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def parallel_write_tfrecord_file(addrs, labels, data_type, image_dims, max_records=sys.maxsize, augment_data=False):
    print("Parallel write tfrecord file {}".format(data_type))
    num_images = min(max_records, len(addrs))
    # Ran into trouble with concurrent.futures using too much memory.  Could make load_image a generator?
    processes = [mp.Process(target=write_tfrecord_file,
                            args=(x, int(num_images / NUM_CPU_CORES * x), int(num_images / NUM_CPU_CORES * (x + 1)),
                                  addrs, labels, data_type, image_dims))
                 for x in range(NUM_CPU_CORES)]
    for p in processes:
        p.start()
    # Exit the completed processes
    for p in processes:
        p.join()
    # Uncomment this to write tfrecord file in single process - good for debugging
    # write_tfrecord_file(0, int(num_images / NUM_CPU_CORES * 0), int(num_images / NUM_CPU_CORES * (0 + 1)),
    #                     addrs, labels, data_type, image_dims)


# Modified from: https://www.dlology.com/blog/how-to-leverage-tensorflows-tfrecord-to-train-keras-model/
def write_tfrecord_file(thread_num, start_index, end_index, addrs, labels, data_type, image_dims):
    filename = 'data/train_val_test_datasets/{}_{}.tfrecords'.format(data_type, thread_num)
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(filename)
    total_files = end_index - start_index
    files_written = 0
    # TODO: Need to provide some sort of indicator of progress (% complete)
    for i in range(start_index, end_index):
        if i % 100 == 0:
            print('Thread {} completed {}/{}'.format(thread_num, i - start_index, total_files))
        augment_data = True if data_type == 'train' else False
        imgs = load_image(addrs[i], augment_data, image_dims)
        for img in imgs:
            label = labels[i]
            feature = {'{}/label'.format(data_type): _bytes_feature(tf.compat.as_bytes(np.asarray(label).tostring())),
                       '{}/image'.format(data_type): _bytes_feature(tf.compat.as_bytes(img.tostring()))}
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
            files_written += 1
    writer.close()
    sys.stdout.flush()
    np.save('data/train_val_test_datasets/{}_{}.npy'.format(data_type, thread_num), np.array([files_written]))


# Modified from: https://www.dlology.com/blog/how-to-leverage-tensorflows-tfrecord-to-train-keras-model/
# TODO: This function is no longer used - Delete?
def generate_tfrecords(cat_dog_train_path, train_frac, val_frac, test_frac):
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
    train_addrs = addrs[0:int(train_frac * len(addrs))]
    train_labels = labels[0:int(train_frac * len(labels))]
    val_addrs = addrs[int(train_frac * len(addrs)):int((train_frac + val_frac) * len(addrs))]
    val_labels = labels[int(train_frac * len(addrs)):int((train_frac + val_frac) * len(addrs))]
    test_addrs = addrs[int((train_frac + val_frac) * len(addrs)):]
    test_labels = labels[int((train_frac + val_frac) * len(labels)):]

    parallel_write_tfrecord_file(train_addrs, train_labels, 'train')
    parallel_write_tfrecord_file(val_addrs, val_labels, 'val')
    parallel_write_tfrecord_file(test_addrs, test_labels, 'test')


def generate_tfrecords_for_image(data_dir, image_dims, train_frac, val_frac, test_frac,
                                 json_to_tensors_fn, class_balancing_fn, max_records=sys.maxsize):
    image_files = glob.glob(os.path.join(data_dir, "*.jpg"))
    image_files.sort()
    image_labels = glob.glob(os.path.join(data_dir, "*.json"))
    image_labels.sort()
    image_label_tensors = []
    for label in image_labels:
        with open(label, 'r') as f:
            image_label_tensors.append(json_to_tensors_fn(json.load(f)))

    # Shuffle the dataset, then put it into an alternating list to ensure equal class representation
    dataset = list(zip(image_files, image_label_tensors))
    shuffle(dataset)
    dataset = dataset[:max_records]
    balanced_images, balanced_label_tensors = class_balancing_fn(dataset)

    train_addrs = balanced_images[0:int(train_frac * len(balanced_images))]
    train_labels = balanced_label_tensors[0:int(train_frac * len(balanced_label_tensors))]
    val_addrs = balanced_images[int(train_frac * len(balanced_images)):
                                int((train_frac + val_frac) * len(balanced_images))]
    val_labels = balanced_label_tensors[int(train_frac * len(balanced_images)):
                                        int((train_frac + val_frac) * len(balanced_images))]
    test_addrs = balanced_images[int((train_frac + val_frac) * len(balanced_images)):]
    test_labels = balanced_label_tensors[int((train_frac + val_frac) * len(balanced_label_tensors)):]
    start_time = time.time()
    parallel_write_tfrecord_file(train_addrs, train_labels, 'train', image_dims)
    cprint("Finished train records in {}".format(time.time() - start_time), "green")
    parallel_write_tfrecord_file(val_addrs, val_labels, 'val', image_dims)
    parallel_write_tfrecord_file(test_addrs, test_labels, 'test', image_dims)

    with open("data/tfrecord_config.json", 'w') as outfile:
            json.dump({"input_dims": image_dims,
                       "output_dims": [len(image_label_tensors[0])],
                       "data_split": [train_frac, val_frac, test_frac]}, outfile)


# Modified from: https://www.dlology.com/blog/how-to-leverage-tensorflows-tfrecord-to-train-keras-model/
def imgs_input_fn(filenames, data_type, input_dims, output_dims, perform_shuffle=False, repeat_count=-1, batch_size=1):
    def _parse_function(serialized):
        features = {'{}/label'.format(data_type): tf.FixedLenFeature([], tf.string),
                    '{}/image'.format(data_type): tf.FixedLenFeature([], tf.string)}
        # Parse the serialized data so we get a dict with our data.
        parsed_example = tf.parse_single_example(serialized=serialized, features=features)
        # Get the image as raw bytes.
        image_raw = parsed_example['{}/image'.format(data_type)]
        label = tf.decode_raw(parsed_example['{}/label'.format(data_type)], tf.int64)
        label = tf.cast(label, tf.float32)
        label = tf.reshape(label, output_dims[1:])
        # Decode the raw bytes so it becomes a tensor with type.
        img = tf.reshape(tf.decode_raw(image_raw, tf.float32), input_dims[1:])
        # Don't know if we need to center the image in this case...
        # image = tf.subtract(image, 116.779) # Zero-center by mean pixel
        return img, label

    dataset = tf.data.TFRecordDataset(filenames=filenames)
    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.
    dataset = dataset.map(map_func=_parse_function, num_parallel_calls=NUM_CPU_CORES)
    if perform_shuffle:
        # Randomizes input using a window of 50 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=50)
    dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times
    dataset = dataset.batch(batch_size)  # Batch size to use
    # How many elements (in this case batches) get consumed per epoch?
    dataset = dataset.prefetch(buffer_size=100)
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


def get_tfrecords(name, base_dir=os.getcwd()):
    records = glob.glob(os.path.join(base_dir, 'data/train_val_test_datasets/{}*.tfrecords'.format(name)))
    numpy_records = glob.glob(os.path.join(base_dir, 'data/train_val_test_datasets/{}*.npy'.format(name)))
    records.sort()
    numpy_records.sort()
    return records, [np.load(x)[0] for x in numpy_records]


def create_val_dir():
    validation_save_path = os.path.join(os.getcwd(), 'data', 'validation_results',
                                        time.strftime("%d_%m_%Y__%H_%M_%S_validation_run"))
    if not os.path.exists(validation_save_path):
        os.makedirs(validation_save_path)
    return validation_save_path


def clean_model_dir(model_dir):
    try:
        shutil.rmtree('data/models/{}/'.format(model_dir))
    except:
        print("Unable to remove directory - perhaps it does not exist?")
    os.makedirs('data/models/{}'.format(model_dir))
    try:
        shutil.rmtree('data/tf_summaries/')
    except:
        print("Unable to remove tf_summaries directory")


def clear_old_tfrecords():
    os.makedirs('data/train_val_test_datasets', exist_ok=True)
    for file in glob.glob('data/train_val_test_datasets/*'):
        os.remove(file)


def clear_dir(graph_dir):
    if os.path.isdir('data/graphs/{}'.format(graph_dir)):
        shutil.rmtree('data/graphs/{}'.format(graph_dir))
    os.makedirs('data/graphs/{}'.format(graph_dir), exist_ok=True)


def read_json_file(path):
    with open(path, 'r') as infile:
        return json.load(infile)


def write_json_file(config_file, output_object):
    with open(config_file, 'w') as outfile:
        json.dump(output_object, outfile)


# Function to turn file names into JSON files for dog and cat dataset
def make_json_from_file_names(data_dir):
    image_files = glob.glob(os.path.join(data_dir, "*.jpg"))
    image_files.sort()
    labels = [[0, 1] if 'cat' in file.split('/')[-1] else [1, 0] for file in image_files]  # 0 = Cat, 1 = Dog
    for ting in zip(labels, image_files):
        original_file_name = ting[1].split('/')[-1]
        new_file_name = ting[1].split('/')[-1].replace("jpg", "json")
        new_file_path = ting[1].replace(original_file_name, new_file_name)
        data = {"label": ting[0]}
        with open(new_file_path, 'w') as outfile:
            json.dump(data, outfile)
