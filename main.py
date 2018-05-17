import tensorflow as tf
import sys

from data_pipeline import generate_tfrecords, imgs_input_fn
from models import cnn_model_fn

def main(argv):
    # Need to set logging verbosity to INFO level or training loss will not print
    tf.logging.set_verbosity(tf.logging.INFO)
    # Training data needs to be split into training, validation, and testing sets
    # This needs to be a complete (not relative) path, or glob will run into issues

    cat_dog_train_path = '/home/michael/Documents/DataSets/dogs_vs_cats_data/*.jpg' if '--laptop' in argv else '/home/michael/hard_drive/datasets/dogs_vs_cats_data/train/*.jpg'
    if '--generate_tfrecords' in sys.argv:
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