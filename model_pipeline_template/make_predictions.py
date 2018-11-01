import re
import os
import cv2
import pdb
import sys
import argparse
import pandas as pd
import tensorflow as tf


from termcolor import cprint
from model_pipeline_utils.data_pipeline import read_json_file
from model_pipeline_utils.train_model_utils import show_image
from model_pipeline_utils.prediction_functions import interactive_check_predictions, get_appropriate_prediction_fn


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


def main(graph_dir, config_file, frozen_graph_names, test_data_dir, output_func, output_labels):
    output_tensor_names = read_json_file("data/model_output_config_{}.json".format(graph_dir))["output"]
    for graph in frozen_graph_names:
        # We use our "load_graph" function
        graph = load_graph('data/graphs/{}/{}.pb'.format(graph_dir, graph))
        # We can verify that we can access the list of operations in the graph
        for op in graph.get_operations():
            print(op.name)
        # We access the input and output nodes
        image_batch = graph.get_tensor_by_name('prefix/image_batch:0')
        is_training = graph.get_tensor_by_name('prefix/is_training/input:0')
        final_dropout_rate = graph.get_tensor_by_name('prefix/final_dropout_rate/input:0')
        input_dims = read_json_file(config_file)["input_dims"]
        preds = graph.get_tensor_by_name('prefix/prediction:0')
        softmax_tensor = graph.get_tensor_by_name('prefix/softmax_tensor:0')
        # Don't need this but oh well
        # softmax_tensor = graph.get_tensor_by_name("prefix/softmax_tensor:0")
        test_predictions, test_ids = [], []
        with tf.Session(graph=graph) as sess:
            # Note: we don't nee to initialize/restore anything
            # There is no Variables in this graph, only hardcoded constants
            for idx, image in enumerate(os.listdir(test_data_dir)):
                # Image gets read in as bgr - need to convert it to rgb
                img = cv2.cvtColor(cv2.imread(os.path.join(test_data_dir, image)), cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (input_dims[1], input_dims[0]))
                img = img / 255.0
                image_with_batch = img.reshape((1,) + img.shape)
                # Cat is 0,1 (s0ftmax tensor), or 1 (predictions)
                test_prediction = sess.run([preds, softmax_tensor], feed_dict={image_batch: image_with_batch, is_training: False, final_dropout_rate: 0})
                test_id = int(re.match(r'\d{1,999999999}', image).group())
                test_predictions.append(test_prediction)
                test_ids.append(test_id)
                print("Finished image {}/{}".format(idx, len(os.listdir(test_data_dir))), end="\r")
                # To interactively go through some predictions, uncomment these lines:
                # interactive_check_predictions(img, prediction)
        prediction_output_fn = get_appropriate_prediction_fn(output_func)
        prediction_output_fn(test_predictions, test_ids, output_labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", default=None, help="Directory graph is stored in -> data/graphs/<GRAPH_DIR>")
    parser.add_argument('--config-file', type=str, default='data/tfrecord_config.json', help='Location of tfrecord_config.json - defaults to the same directory as train_model.py')
    parser.add_argument("--frozen-graph-names", default="data/graphs/frozen_model.pb", type=str, help="Frozen model file(s) to import -> data/graphs/<GRAPH_DIR>/<GRAPH_NAME,GRAPH_NAME,...>")
    parser.add_argument("--test-data-dir", default=None, help="Full path to test data directory")
    parser.add_argument("--output-func", default=None, help="Name of function to create prediction(s)")
    parser.add_argument("--output-labels", default=None, help="Name of the keys in the prediction csv(s)/json(s)")
    args = parser.parse_args()
    main(args.graph_dir, args.config_file, args.frozen_graph_names.split(","), args.test_data_dir, args.output_func, args.output_labels.split(','))

# Example: python3 make_predictions.py --graph-dir cat_dog_cnn_desktop --frozen-graph-names model_10,model_12 --test-data-dir '/home/michael/hard_drive/datasets/dogs_vs_cats_data/test' --output-func cat_dog_classifier_output --output-labels id,label