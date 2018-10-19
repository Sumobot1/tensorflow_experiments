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
import shutil

from functools import reduce
from model_pipeline_utils.data_pipeline import generate_tfrecords, imgs_input_fn, get_tfrecords, clear_old_tfrecords, clean_model_dir, create_val_dir, clear_dir
from model_pipeline_utils.models import cnn_model_fn, fast_cnn_model_fn
from model_pipeline_utils.train_model_utils import average, get_num_steps, train_model, get_appropriate_model, read_model_config, get_io_placeholders
import argparse

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


def main(graph_dir, config_file, model_dir, model_files):
    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.set_verbosity(tf.logging.WARN)
    for model_file in model_files:
        graph_dir = 'graphs/{}'.format(graph_dir)
        clear_dir(graph_dir)
        model_path = 'models/{}/{}'.format(model_dir, model_file)
        # Model path should look something like this: 'models/cat_dog_cnn_desktop/model_13'
        if not tf.train.checkpoint_exists(model_path):
            print("Checkpoint {} does not exist.  Skipping...".format(model_path))
            continue
        # validation_save_path = create_val_dir()
        graph_path = 'graphs/{}/{}.pb'.format(graph_dir, model_file) if model_file else None
        with tf.Session(graph=tf.Graph()) as sess:
            # We import the meta graph in the current default Graph
            saver = tf.train.import_meta_graph(model_path + '.meta', clear_devices=True)
            # We restore the weights
            saver.restore(sess, model_path)
            output_node_names = ["prediction", "softmax_tensor"]
            # We use a built-in TF helper to export variables to constants
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,
                tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
                output_node_names#.split(",") # The output node names are used to select the usefull nodes
            )

            # Finally we serialize and dump the output graph to the filesystem
            with tf.gfile.GFile(graph_path, "wb") as f:
                f.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph." % len(output_graph_def.node))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph-dir', type=str, default=None, help="Directory to store exported graphs")
    parser.add_argument('--config-file', type=str, default='tfrecord_config.json', help='Location of tfrecord_config.json - defaults to the same directory as train_model.py')
    parser.add_argument('--model-dir', type=str, default=None, help="Directory to store model checkpoints in/load model checkpoints from")
    parser.add_argument('model_files', nargs='?', default=None)
    args = parser.parse_args()
    main(args.graph_dir, args.config_file, args.model_dir, args.model_files.split(' '))

# Example Usage: python3 export_model.py --graph-dir cat_dog_cnn_desktop --config-file tfrecord_config.json --model-dir cat_dog_cnn_desktop model_4
