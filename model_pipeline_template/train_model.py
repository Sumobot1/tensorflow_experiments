import tensorflow as tf
import sys
import pdb
import multiprocessing as mp
import glob
import shutil
import os
import numpy as np
import time
import json

from functools import reduce
from model_pipeline_utils.data_pipeline import generate_tfrecords, imgs_input_fn, get_tfrecords, clear_old_tfrecords, clean_model_dir, create_val_dir
from model_pipeline_utils.models import cnn_model_fn, fast_cnn_model_fn
from model_pipeline_utils.train_model_utils import average, get_num_steps, train_model, get_appropriate_model, read_model_config
import argparse

NUM_EPOCHS = 80
DATA_REPETITIONS_PER_EPOCH = 1
VAL_BATCH_SIZE = 300

# Notes:
# 1. In general it is considered good practice to use list comprehension instead of map 99% of the time.
# 2, If values are being printed using logging.info, need to set logging verbosity to INFO level or training loss will not print
# 3, Training data needs to be split into training, validation, and testing sets
# 4, Glob seems to need a complete (not relative) path, or it will run into issues?  Might want to test this again
# 5. A good way to debug programs is to run a tf.InteractiveSession()
# 6. Tensorflow importing datasets: https://www.tensorflow.org/programmers_guide/datasets
# 7. Random stuff on protobuf's queues: https://indico.io/tensorflow-data-inputs-part1-placeholders-protobufs-queues/
# 8. Dataset.repeat(-1) - this causes the dataset to repeat indefinitely
# 9. It does not appear to be possible to disable the histogram summary and reload the model at a different checkpoint.


def main(clean_dir, num_epochs, val_start_epoch, summary_start_epoch, train_batch_size, config_file, model_dir, model_name, model_file):
    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.set_verbosity(tf.logging.WARN)
    validation_save_path = create_val_dir()
    ckpt_path = 'models/{}/{}'.format(model_dir, model_file) if model_file else None
    if clean_dir:
        clean_model_dir()
    train_frac, val_frac, test_frac, input_dims, output_dims = read_model_config(config_file)
    # sess = tf.InteractiveSession()
    # next_example, next_label = imgs_input_fn(['train_0.tfrecords'], 'train', perform_shuffle=True, repeat_count=5, batch_size=20)
    train_records, train_record_lengths = get_tfrecords('train')
    val_records, val_record_lengths = get_tfrecords('val')
    total_training_files = sum(train_record_lengths)
    total_num_steps = int(total_training_files / train_batch_size)
    print("NUM_ROTATIONS: {}, TOTAL TRAINING FILES: {}, TOTAL NUM STEPS {}".format(1, total_training_files, total_num_steps))
    model_fn = get_appropriate_model(model_name)

    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    next_example, next_label = imgs_input_fn(train_records, 'train', perform_shuffle=True, repeat_count=-1, batch_size=train_batch_size)
    next_val_example, next_val_label = imgs_input_fn(val_records, 'val', perform_shuffle=False, repeat_count=-1, batch_size=VAL_BATCH_SIZE)
    image_batch = tf.placeholder_with_default(next_example, shape=input_dims)
    label_batch = tf.placeholder_with_default(next_label, shape=output_dims)
    image_val_batch = tf.placeholder_with_default(next_val_example, shape=input_dims)
    label_val_batch = tf.placeholder_with_default(next_val_label, shape=output_dims)
    # Cannot change histogram summary and then reload model from the same checkpoint
    loss, predictions = model_fn(image_batch, label_batch, mode=tf.estimator.ModeKeys.TRAIN, params={"return_estimator": False, "total_num_steps": total_num_steps, "histogram_summary": False, "loss_summary": True, "show_graph": True})
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss, name="training_op")
    num_steps = get_num_steps(train_record_lengths, train_batch_size, DATA_REPETITIONS_PER_EPOCH)
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
    parser.add_argument('--clean', action="store_true", default=False, help='Get rid of any models in model directory')
    parser.add_argument('--num-epochs', type=int, default=0, help='Number of epochs to train for')
    parser.add_argument('--start-val-at', type=int, default=0, help='Do not test against the validation set until a certain epoch')
    parser.add_argument('--start-summary-at', type=int, default=0, help='Do not start the summary until a certain epoch')
    parser.add_argument('--train-batch-size', type=int, default=1, help="Training batch size")
    parser.add_argument('--config-file', type=str, default='tfrecord_config.json', help='Location of tfrecord_config.json - defaults to the same directory as train_model.py')
    parser.add_argument('--model-dir', type=str, default=None, help="Directory to store model checkpoints in/load model checkpoints from")
    parser.add_argument('--model-name', type=str, default='cnn_model_fn', help="Name of model function - should match a model in model_pipeline_utils.models")
    parser.add_argument('model_file', nargs='?', default=None)
    args = parser.parse_args()
    main(args.clean, args.num_epochs, args.start_val_at, args.start_summary_at, args.train_batch_size, args.config_file, args.model_dir, args.model_name, args.model_file)

# Example Usage: 