from functools import reduce
import tensorflow as tf
import time
import numpy as np
import os


def average(list):
    return reduce(lambda x, y: x + y, list) / len(list)


def get_num_steps(records_array, batch_size, repititions_per_epoch=1):
    return int(sum(records_array) * repititions_per_epoch / batch_size)


def train_model(sess, num_steps, num_epochs, image_batch, label_batch, loss, predictions, training_op, num_val_steps, image_val_batch, label_val_batch, validation_save_path):
    sess.run(tf.global_variables_initializer())
    for epoch in range(num_epochs):
        # TRAINING
        start = time.time()
        for step in range(num_steps):
            X, Y = sess.run([image_batch, label_batch])
            cost_value, predictions_value, _ = sess.run([loss, predictions, training_op], feed_dict={image_batch: X, label_batch: Y})
            # Note: Do NOT add accuracy calculation here.  It makes training much slower! (6s vs 19s)
            # correct = tf.equal(tf.argmax(input=Y, axis=1), predictions_value["classes"], name="correct")
            # accuracy = sess.run(tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy"))
            print("Step {} complete, cost: {:0.5f}".format(step, cost_value), end="\r")
        print()
        print("Time: {} - {} seconds per step".format(time.time() - start, float(time.time() - start) / float(num_steps)))
        # VALIDATION
        x_val = None
        y_val = None
        y_pred_val = None
        val_cost = 0
        start_ting = time.time()
        for step in range(num_val_steps):
            # DO i NEED TO SESS.RUN THIS EVERY STEP?  CAN i JUST DO IT OUTSIDE OF THE STEPS/EPOCHS?
            X_val, Y_val = sess.run([image_val_batch, label_val_batch])
            # Need to send loss, predictions (outputs from cnn_model_fn) above.  Need to use same cnn model function for both training and validation sets
            cost_val_value, y_val_pred = sess.run([loss, predictions], feed_dict={image_batch: X_val, label_batch: Y_val})
            x_val = X_val if x_val is None else np.concatenate((x_val, X_val))
            y_val = Y_val if y_val is None else np.concatenate((y_val, Y_val))
            y_pred_val = y_val_pred['probabilities'] if y_pred_val is None else np.concatenate((y_pred_val, y_val_pred['probabilities']))
            val_cost += cost_val_value
            print("Val Step Complete, cost: {:0.5f}".format(cost_val_value), end="\r")
        print()
        print("Done - Time: {} avg validation cost: {}".format(time.time() - start_ting, val_cost / float(num_val_steps)))
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