import tensorflow as tf
import sys
import pdb
import multiprocessing as mp
import glob
import shutil
import os
import numpy as np
import time

from functools import reduce
from data_pipeline import generate_tfrecords, imgs_input_fn, get_tfrecords, clear_old_tfrecords, clean_model_dir, create_val_dir
from models import cnn_model_fn, fast_cnn_model_fn
from utils import average, get_num_steps, train_model

NUM_EPOCHS = 5
DATA_REPETITIONS_PER_EPOCH = 6
VAL_BATCH_SIZE = 300


def main(argv):
    machine_type = 'laptop' if '--laptop' in argv else 'desktop'
    # If values are being printed using logging.info, need to set logging verbosity to INFO level or training loss will not print
    # I'm using a custom printing function, so this does not need to be done
    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.set_verbosity(tf.logging.WARN)
    # Training data needs to be split into training, validation, and testing sets
    # This needs to be a complete (not relative) path, or glob will run into issues
    cat_dog_train_path = '/home/michael/Documents/DataSets/dogs_vs_cats_data/*.jpg' if machine_type == 'laptop' else '/home/michael/hard_drive/datasets/dogs_vs_cats_data/train/*.jpg'
    if '--generate_tfrecords' in sys.argv:
        clear_old_tfrecords()
        generate_tfrecords(cat_dog_train_path)
    if '--clean' in sys.argv:
        clean_model_dir()

    validation_save_path = create_val_dir()
    # A good way to debug programs like this is to run a tf.InteractiveSession()
    # sess = tf.InteractiveSession()
    # next_example, next_label = imgs_input_fn(['train_0.tfrecords'], 'train', perform_shuffle=True, repeat_count=5, batch_size=20)

    training_batch_size = 1 if machine_type == 'laptop' else 20
    train_records, train_record_lengths = get_tfrecords('train')
    # In general it is considered good practice to use list comprehension instead of map 99% of the time.
    val_records, val_record_lengths = get_tfrecords('val')

    # Steps is how many times to call next on the input function - ie how many batches to take in?
    repeat_count = 5
    # Multiplied by 0.6 because training files are 60% of data
    total_training_files = int(len(glob.glob(cat_dog_train_path)) * 0.6) * repeat_count
    total_num_steps = int(total_training_files / training_batch_size)
    print("TOTAL FILES: {}, NUM_ROTATIONS: {}, TOTAL TRAINING FILES: {}, TOTAL NUM STEPS {}".format(len(cat_dog_train_path), 1, total_training_files, total_num_steps))
    model_fn = fast_cnn_model_fn if machine_type == 'laptop' else cnn_model_fn
    # New Code to Read Stuff Inside of a Session ==========================================================================================================================
    # Tensorflow importing datasets: https://www.tensorflow.org/programmers_guide/datasets
    # Random shit on protobuf's queues: https://indico.io/tensorflow-data-inputs-part1-placeholders-protobufs-queues/
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    next_example, next_label = imgs_input_fn(['train_0.tfrecords'], 'train', perform_shuffle=True, repeat_count=NUM_EPOCHS * DATA_REPETITIONS_PER_EPOCH, batch_size=training_batch_size)
    next_val_example, next_val_label = imgs_input_fn(val_records, 'val', perform_shuffle=False, repeat_count=NUM_EPOCHS, batch_size=VAL_BATCH_SIZE)
    image_batch = tf.placeholder_with_default(next_example, shape=[None, 80, 80, 3])
    label_batch = tf.placeholder_with_default(next_label, shape=[None, 2])
    image_val_batch = tf.placeholder_with_default(next_val_example, shape=[None, 80, 80, 3])
    label_val_batch = tf.placeholder_with_default(next_val_label, shape=[None, 2])
    loss, predictions = model_fn(image_batch, label_batch, mode=tf.estimator.ModeKeys.TRAIN, params={"return_estimator": False, "total_num_steps": total_num_steps})
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss, name="training_op")
    print("done, {}".format(train_record_lengths))
    num_steps = get_num_steps([train_record_lengths[0]], training_batch_size, DATA_REPETITIONS_PER_EPOCH)
    print("done, {}".format(val_record_lengths))
    num_val_steps = get_num_steps(val_record_lengths, VAL_BATCH_SIZE, 1)
    train_model(sess, num_steps, NUM_EPOCHS, image_batch, label_batch, loss, predictions, training_op, num_val_steps, image_val_batch, label_val_batch, validation_save_path)


if __name__ == "__main__":
    main(sys.argv)
