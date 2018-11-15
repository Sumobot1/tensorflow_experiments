import tensorflow as tf
from termcolor import cprint
from model_pipeline_utils.data_pipeline import clear_dir, write_json_file
import argparse

DATA_REPETITIONS_PER_EPOCH = 1
VAL_BATCH_SIZE = 300

# Notes:
# 1. In general it is considered good practice to use list comprehension instead of map 99% of the time.
# 2, If values are being printed using logging.info, need to set logging verbosity to INFO level
#    or training loss will not print
# 3, Training data needs to be split into training, validation, and testing sets
# 4, Glob seems to need a complete (not relative) path, or it will run into issues?  Might want to test this again
# 5. A good way to debug programs is to run a tf.InteractiveSession()
# 6. Tensorflow importing datasets: https://www.tensorflow.org/programmers_guide/datasets
# 7. Random stuff on protobuf's queues: https://indico.io/tensorflow-data-inputs-part1-placeholders-protobufs-queues/
# 8. Dataset.repeat(-1) - this causes the dataset to repeat indefinitely
# 9. It does not appear to be possible to disable the histogram summary and reload the model at a different checkpoint.


def main(graph_dir, config_file, model_dir, output_node_names, model_files):
    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.set_verbosity(tf.logging.WARN)
    clear_dir(graph_dir)
    for model_file in model_files:
        model_path = 'data/models/{}/{}'.format(model_dir, model_file)
        # Model path should look something like this: 'models/cat_dog_cnn_desktop/model_13'
        if not tf.train.checkpoint_exists(model_path):
            cprint("Checkpoint {} does not exist.  Skipping...".format(model_path), 'yellow')
            continue
        graph_path = 'data/graphs/{}/{}.pb'.format(graph_dir, model_file) if model_file else None
        with tf.Session(graph=tf.Graph()) as sess:
            # We import the meta graph in the current default Graph
            saver = tf.train.import_meta_graph(model_path + '.meta', clear_devices=True)
            # We restore the weights
            saver.restore(sess, model_path)
            # We use a built-in TF helper to export variables to constants
            # The graph_def is used to retrieve the nodes
            # The output node names are used to select the usefull nodes
            output_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                                                            tf.get_default_graph().as_graph_def(),
                                                                            output_node_names)
            # Finally we serialize and dump the output graph to the filesystem
            with tf.gfile.GFile(graph_path, "wb") as f:
                f.write(output_graph_def.SerializeToString())
            cprint("Finished saving {}... {} ops in the final graph."
                   .format(model_file, len(output_graph_def.node)), 'green')
    write_json_file("data/model_output_config_{}.json".format(graph_dir), {"output": output_node_names})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph-dir', type=str, default=None,
                        help="Directory to store exported graphs -> data/graphs/<GRAPH_DIR>")
    parser.add_argument('--config-file', type=str, default='tfrecord_config.json',
                        help='Location of tfrecord_config.json - defaults to the same directory as train_model.py')
    parser.add_argument('--model-dir', type=str, default=None,
                        help="Directory to store model checkpoints in/load model checkpoints from:"
                             "-> data/models/<MODEL_DIR>")
    parser.add_argument('--output-node-names', type=str, default="prediction,softmax_tensor",
                        help='Names of output nodes.'
                             'These will be declared in the model file - Defaults to "prediction,softmax_tensor"')
    parser.add_argument('model_files', nargs='?', default=None,
                        help="Names of model files -> data/models/<MODEL_DIR>/<MODEL_FILE,MODEL_FILE,...>")
    args = parser.parse_args()
    main(args.graph_dir, args.config_file, args.model_dir, args.output_node_names.split(","), args.model_files.split(','))

# Example Usage: python3 export_model.py --graph-dir cat_dog_cnn_desktop --config-file tfrecord_config.json --model-dir cat_dog_cnn_desktop model_4
