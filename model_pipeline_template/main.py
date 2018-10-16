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
from model_pipeline_utils.data_pipeline import generate_tfrecords, imgs_input_fn, get_tfrecords, clear_old_tfrecords, clean_model_dir, create_val_dir
from model_pipeline_utils.models import cnn_model_fn, fast_cnn_model_fn
from model_pipeline_utils.train_model_utils import average, get_num_steps, train_model
import argparse

NUM_EPOCHS = 80
DATA_REPETITIONS_PER_EPOCH = 1
VAL_BATCH_SIZE = 300


def main(clean_dir, gen_records, is_laptop, num_epochs, val_start_epoch, summary_start_epoch, train_val_test_split, model_file):
    if len(train_val_test_split) != 3 or sum(train_val_test_split) != 1:
        print("ERROR - Train + Val + Test should equal 1")
        return
    # In general it is considered good practice to use list comprehension instead of map 99% of the time.
    # If values are being printed using logging.info, need to set logging verbosity to INFO level or training loss will not print
    # I'm using a custom printing function, so this does not need to be done
    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.set_verbosity(tf.logging.WARN)
    # Training data needs to be split into training, validation, and testing sets
    # This needs to be a complete (not relative) path, or glob will run into issues
    train_frac, val_frac, test_frac = train_val_test_split
    cat_dog_train_path = '/home/michael/Documents/DataSets/dogs_vs_cats_data/*.jpg' if is_laptop else '/home/michael/hard_drive/datasets/dogs_vs_cats_data/train/*.jpg'
    training_batch_size = 1 if is_laptop else 110
    validation_save_path = create_val_dir()
    model_dir = 'cat_dog_cnn_laptop' if is_laptop else 'cat_dog_cnn_desktop'
    ckpt_path = None
    if model_file:
        ckpt_path = 'models/{}/{}'.format(model_dir, model_file) if is_laptop else 'models/{}/{}'.format(model_dir, model_file)

    if gen_records:
        clear_old_tfrecords()
        generate_tfrecords(cat_dog_train_path, train_frac, val_frac, test_frac)
    if clean_dir:
        clean_model_dir()

    # A good way to debug programs like this is to run a tf.InteractiveSession()
    # sess = tf.InteractiveSession()
    # next_example, next_label = imgs_input_fn(['train_0.tfrecords'], 'train', perform_shuffle=True, repeat_count=5, batch_size=20)

    train_records, train_record_lengths = get_tfrecords('train')
    val_records, val_record_lengths = get_tfrecords('val')
    # Multiplied by 0.6 because training files are 60% of data
    total_training_files = int(len(glob.glob(cat_dog_train_path)) * train_frac)
    total_num_steps = int(total_training_files / training_batch_size)
    print("TOTAL FILES: {}, NUM_ROTATIONS: {}, TOTAL TRAINING FILES: {}, TOTAL NUM STEPS {}".format(len(cat_dog_train_path), 1, total_training_files, total_num_steps))
    model_fn = fast_cnn_model_fn if is_laptop else cnn_model_fn
    # Tensorflow importing datasets: https://www.tensorflow.org/programmers_guide/datasets
    # Random shit on protobuf's queues: https://indico.io/tensorflow-data-inputs-part1-placeholders-protobufs-queues/
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    # repeat_count=-1 repeats the dataset indefinitely
    next_example, next_label = imgs_input_fn(train_records, 'train', perform_shuffle=True, repeat_count=-1, batch_size=training_batch_size)
    next_val_example, next_val_label = imgs_input_fn(val_records, 'val', perform_shuffle=False, repeat_count=-1, batch_size=VAL_BATCH_SIZE)
    # Prob going to want to read things like input sizes from a config file (keep things consistent between preparing data, and training the network)
    image_batch = tf.placeholder_with_default(next_example, shape=[None, 80, 80, 3])
    label_batch = tf.placeholder_with_default(next_label, shape=[None, 2])
    image_val_batch = tf.placeholder_with_default(next_val_example, shape=[None, 80, 80, 3])
    label_val_batch = tf.placeholder_with_default(next_val_label, shape=[None, 2])
    # Cannot change histogram summary and then reload model from the same checkpoint
    loss, predictions = model_fn(image_batch, label_batch, mode=tf.estimator.ModeKeys.TRAIN, params={"return_estimator": False, "total_num_steps": total_num_steps, "histogram_summary": False, "loss_summary": True, "show_graph": True})
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss, name="training_op")
    num_steps = get_num_steps(train_record_lengths, training_batch_size, DATA_REPETITIONS_PER_EPOCH)
    print("Train record lengths: {}".format(train_record_lengths))
    print("Val record lengths: {}".format(val_record_lengths))
    num_val_steps = get_num_steps(val_record_lengths, VAL_BATCH_SIZE, 1)
    os.makedirs("tf_summaries/train", exist_ok=True)
    os.makedirs("tf_summaries/val", exist_ok=True)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('tf_summaries/train', sess.graph)
    test_writer = tf.summary.FileWriter('tf_summaries/val')
    print("num_steps: {}".format(num_val_steps))
    train_model(sess, num_steps, num_epochs, image_batch, label_batch, loss, predictions, training_op, num_val_steps, image_val_batch, label_val_batch, validation_save_path, merged, train_writer, test_writer, ckpt_path, model_dir, val_start_epoch, summary_start_epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--laptop', action="store_true", default=False, help='Run a smaller model if on laptop - otherwise training will take forever')
    parser.add_argument('--generate-tfrecords', action="store_true", default=False, help='Regenerate tfrecord files if we have changed preprocessing')
    parser.add_argument('--clean', action="store_true", default=False, help='Get rid of any models in model directory')
    parser.add_argument('--num-epochs', type=int, default=0, help='Number of epochs to train for')
    parser.add_argument('--start-val-at', type=int, default=0, help='Do not test against the validation set until a certain epoch')
    parser.add_argument('--start-summary-at', type=int, default=0, help='Do not start the summary until a certain epoch')
    parser.add_argument('--train-val-test-split', type=str, help="Train val test split - should add up to 1.  Should be of form '<FLOAT>, <FLOAT>, <FLOAT>'")
    parser.add_argument('model_file', nargs='?', default=None)
    args = parser.parse_args()
    main(args.clean, args.generate_tfrecords, args.laptop, args.num_epochs, args.start_val_at, args.start_summary_at, [float(item) for item in args.train_val_test_split.split(', ')], args.model_file)
