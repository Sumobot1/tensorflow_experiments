import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pdb
import json
from functools import reduce
import tensorflow as tf
import time
import numpy as np
import os
import sys
from termcolor import cprint
from model_pipeline_utils.models import fast_cnn_model_fn, cnn_model_fn


def show_image(img):
    # cv2.imshow('image',img)
    plt.imshow(img, interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


def average(list):
    return reduce(lambda x, y: x + y, list) / len(list)


def get_num_steps(records_array, batch_size, repititions_per_epoch=1):
    return int(sum(records_array) * repititions_per_epoch / batch_size)


def get_appropriate_model(model_name):
    model_dict = {"fast_cnn_model_fn": fast_cnn_model_fn,
                  "cnn_model_fn": cnn_model_fn}
    return model_dict[model_name]


def train_model_step(sess, session_vars, session_dict, epoch, epochs_before_summary, summary_writer, global_step, session_type):
    cost_value = None
    predictions_value = None
    session_vals = sess.run(session_vars, feed_dict=session_dict)
    if epoch >= epochs_before_summary:
        summary, cost_value, predictions_value = session_vals[0], session_vals[1], session_vals[2]
        summary_writer.add_summary(summary, global_step)
    else:
        cost_value, predictions_value = session_vals[0], session_vals[1]
    return cost_value, predictions_value


def read_model_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
        input_dims = [None] + config['input_dims']
        output_dims = [None] + config['output_dims']
        train_frac, val_frac, test_frac = config['data_split']
        return train_frac, val_frac, test_frac, input_dims, output_dims


def get_io_placeholders(next_example, next_label, input_dims, output_dims, input_name, output_name):
    return tf.placeholder_with_default(next_example, shape=input_dims, name=input_name), tf.placeholder_with_default(next_label, shape=output_dims, name=output_name)


def train_model(sess, num_steps, num_epochs, image_batch, label_batch, loss, predictions, training_op, num_val_steps, image_val_batch, label_val_batch, validation_save_path, merged, train_writer, test_writer, ckpt_path, model_dir, epochs_before_validation, epochs_before_summary, is_train, final_dropout_rate):
    saver = tf.train.Saver(max_to_keep=num_epochs)
    starting_epoch, counter = 0, 0
    min_validation_cost = sys.maxsize
    # Note - the model checkpoints only have the weights - the model still needs to be declared
    if ckpt_path is not None and tf.train.checkpoint_exists(ckpt_path):
        print("Restoring checkpoint {}".format(ckpt_path))
        saver.restore(sess, ckpt_path)
        starting_epoch = int(ckpt_path.split('_')[-1]) + 1
        min_validation_cost = float(np.load('data/models/{}/min_cost.npy'.format(model_dir)))
    else:
        print("No checkpoint to restore... starting from scratch")
        sess.run(tf.global_variables_initializer())
    for epoch in range(starting_epoch, num_epochs):
        train_start = time.time()
        for step in range(num_steps):
            X, Y = sess.run([image_batch, label_batch])
            # Image is in RGB format (Not BGR)
            session_vars = [merged, loss, predictions, training_op] if epoch >= epochs_before_summary else [loss, predictions, training_op]
            cost_value, predictions_value = train_model_step(sess, session_vars, {image_batch: X, label_batch: Y, is_train: True, final_dropout_rate: 0.9}, epoch, epochs_before_summary, train_writer, counter, "train")
            counter += 1
            # Note: Do NOT add accuracy calculation here.  It makes training much slower! (6s vs 19s)
            # correct = tf.equal(tf.argmax(input=Y, axis=1), predictions_value["classes"], name="correct")
            # accuracy = sess.run(tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy"))
            print("Step {} complete, cost: {:0.5f}".format(step, cost_value), end="\r")
        print("\nTime: {} - {} seconds per step".format(time.time() - train_start, float(time.time() - train_start) / float(num_steps)))
        if epoch >= epochs_before_validation:
            x_val, y_val, y_pred_val = None, None, None
            val_cost = 0
            val_start = time.time()
            for step in range(num_val_steps):
                X_val, Y_val = sess.run([image_val_batch, label_val_batch])
                # Need to send loss, predictions (outputs from cnn_model_fn) above.  Need to use same cnn model function for both training and validation sets
                # https://www.tensorflow.org/performance/performance_guide#general_best_practices
                # ^ Feed dict is not much slower than the tf.data api for a single gpu
                session_vars = [merged, loss, predictions] if epoch >= epochs_before_summary else [loss, predictions]
                cost_val_value, y_val_pred = train_model_step(sess, session_vars, {image_batch: X_val, label_batch: Y_val, is_train: True, final_dropout_rate: 0}, epoch, epochs_before_summary, test_writer, counter, "test")
                counter += 1
                x_val = X_val if x_val is None else np.concatenate((x_val, X_val))
                y_val = Y_val if y_val is None else np.concatenate((y_val, Y_val))
                y_pred_val = y_val_pred['probabilities'] if y_pred_val is None else np.concatenate((y_pred_val, y_val_pred['probabilities']))
                val_cost += cost_val_value
                print("Val Step Complete, cost: {:0.5f}".format(cost_val_value), end="\r")
            avg_val_cost = val_cost / float(num_val_steps)
            cprint("\nDone - Time: {} avg validation cost: {}".format(time.time() - val_start, avg_val_cost), "green")
            if min_validation_cost > avg_val_cost:
                save_path = saver.save(sess, "data/models/{}/model_{}".format(model_dir, epoch))
                min_validation_cost = avg_val_cost
                np.save("data/models/{}/min_cost.npy".format(model_dir), np.array(min_validation_cost))
                print("Saved validation at {}".format(save_path))
            ckpt_path = os.path.join(validation_save_path, 'epoch_{}'.format(epoch))
            os.mkdir(ckpt_path)
            np.save(os.path.join(ckpt_path, "x_val.npy"), x_val)
            np.save(os.path.join(ckpt_path, "y_val.npy"), y_val)
            np.save(os.path.join(ckpt_path, "y_pred_val.npy"), y_pred_val)
            print("File saved at checkpoint path {}".format(ckpt_path))

    # # Current (Working) Estimator Code ==========================================================================================================================
    # if not os.path.exists('models'):
    #     os.makedirs('models')
    # mnist_classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir="models/cat_dog_cnn_{}".format(machine_type), params={'total_num_steps': total_num_steps})
    # for i in range(10):
    #     print("Latest checkpoint =====================: {}".format(mnist_classifier.latest_checkpoint()))
    #     mnist_classifier.train(input_fn=lambda: imgs_input_fn(train_records, 'train', perform_shuffle=True, repeat_count=2, batch_size=training_batch_size), steps=total_num_steps)
    #     eval_results = mnist_classifier.evaluate(input_fn=lambda: imgs_input_fn(val_records, 'val', perform_shuffle=False, repeat_count=1))
    #     print(eval_results)
    # # ===========================================================================================================================================================