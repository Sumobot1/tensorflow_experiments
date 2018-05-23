import tensorflow as tf
import sys
import pdb
import multiprocessing as mp
import glob
import os
from data_pipeline import generate_tfrecords, imgs_input_fn
from models import cnn_model_fn, fast_cnn_model_fn
from estimator_hooks import SuperWackHook

def get_tfrecords(name):
    records = glob.glob('{}*.tfrecords'.format(name))
    records.sort()
    return records


def main(argv):
    machine_type = 'laptop' if '--laptop' in argv else 'desktop'
    # Need to set logging verbosity to INFO level or training loss will not print
    tf.logging.set_verbosity(tf.logging.INFO)
    # Training data needs to be split into training, validation, and testing sets
    # This needs to be a complete (not relative) path, or glob will run into issues

    cat_dog_train_path = '/home/michael/Documents/DataSets/dogs_vs_cats_data/*.jpg' if machine_type == 'laptop' else '/home/michael/hard_drive/datasets/dogs_vs_cats_data/train/*.jpg'
    if '--generate_tfrecords' in sys.argv:
        for file in glob.glob('*.tfrecords'):
            os.remove(file)
        generate_tfrecords(cat_dog_train_path)

    next_example, next_label = imgs_input_fn(['train.tfrecords'], 'train', perform_shuffle=True, repeat_count=5, batch_size=20)
    
    # A good way to debug programs like this is to run a tf.InteractiveSession()
    # sess = tf.InteractiveSession()
    model_fn = fast_cnn_model_fn if machine_type == 'laptop' else cnn_model_fn
    if not os.path.exists('models'):
        os.makedirs('models')
    mnist_classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir="models/cat_dog_cnn_{}".format(machine_type))
    tensors_to_log = {"probabilities": "ting"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
    wack_hook = SuperWackHook(tensors=tensors_to_log, every_n_iter=50)
    training_batch_size = 1 if machine_type == 'laptop' else 20
    train_records = get_tfrecords('train')
    val_records = get_tfrecords('val')
    # glob.glob('train*.tfrecords')
    # train_records.sort()


    # Steps is how many times to call next on the input function - ie how many batches to take in?
    repeat_count = 1
    total_training_files = len(glob.glob(cat_dog_train_path)) * 1 + len(cat_dog_train_path)
    total_num_steps = int(total_training_files / training_batch_size * repeat_count)
    print("TOTAL FILES: {}, NUM_ROTATIONS: {}, TOTAL TRAINING FILES: {}, TOTAL NUM STEPS {}".format(len(cat_dog_train_path), 1, total_training_files, total_num_steps))
    for i in range(5):
        # logging_hook
        hooks = [wack_hook]
        mnist_classifier.train(input_fn=lambda: imgs_input_fn(train_records, 'train', perform_shuffle=True, repeat_count=1, batch_size=training_batch_size), steps=total_num_steps, hooks=hooks)
        eval_results = mnist_classifier.evaluate(input_fn=lambda: imgs_input_fn(val_records, 'val', perform_shuffle=False, repeat_count=1))
        print(eval_results)
if __name__ == "__main__":
    main(sys.argv)