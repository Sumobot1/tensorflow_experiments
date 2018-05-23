import tensorflow as tf
import sys
import pdb
import multiprocessing as mp
import glob
import os
from data_pipeline import generate_tfrecords, imgs_input_fn
from models import cnn_model_fn, fast_cnn_model_fn

from tensorflow.python.training import session_run_hook
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.framework import ops
import six
import numpy as np
from tensorflow.python.platform import tf_logging as logging
# # @tf_export("train.SuperWackHook")
# class SuperWackHook(session_run_hook.SessionRunHook):
# # class SuperWackHook(tf.train.SessionRunHook):
#     """Saves summaries during eval loop."""

#     def __init__(self,
#                  tensors=None,
#                  every_n_iter=None):
#         """Initializes a special `SummarySaverHook` to run during evaluations

#         Args:
#           output_dir: `string`, the directory to save the summaries to.
#         """
#         # self._summary_op = None
#         # self._output_dir = output_dir
#         # self._global_step_tensor = None
#         # self._stop_after = stop_after
#         # self._saves = None

#     def begin(self):
#         print("begin")
#         # self._global_step_tensor = tf.train.get_or_create_global_step()
#         # if self._global_step_tensor is None:
#         #     raise RuntimeError(
#         #         "Global step should be created to use SummarySaverHook.")

#     def before_run(self, run_context):  # pylint: disable=unused-argument
#         print("before_run")
#         # requests = {"global_step": self._global_step_tensor}
#         # if self._saves is None: # skip first time, because summaries only appear after first run
#         #     self._saves = 0
#         # elif self._saves < self._stop_after and self._get_summary_op() is not None:
#         #     requests["summary"] = self._get_summary_op()

#         # return tf.train.SessionRunArgs(requests)

#     def after_run(self, run_context, run_values):
#         _ = run_context
#         print('run_context: {}'.format(run_context))
#         print("after run: {}".format(run_values.results))
#         print("run_values: {}".format(run_values))
#         pdb.set_trace()
#         # _ = run_context
#         # if "summary" in run_values.results:
#         #     print('Saving eval summaries')
#         #     global_step = run_values.results["global_step"]
#         #     summary_writer = tf.summary.FileWriterCache.get(self._output_dir)

#         #     for summary in run_values.results["summary"]:
#         #         summary_writer.add_summary(summary, global_step)

#         #     self._saves += 1

#     def end(self, session=None):
#         print('end')
#         # _ = session
#         # summary_writer = tf.summary.FileWriterCache.get(self._output_dir)
#         # summary_writer.flush()

#     def _get_summary_op(self):
#         """Fetches the summary op from collections
#         """

#         if self._summary_op is None:
#             self._summary_op = tf.get_collection(tf.GraphKeys.SUMMARY_OP)

#         return self._summary_op
# @tf_export("train.LoggingTensorHook")
def _as_graph_element(obj):
  """Retrieves Graph element."""
  graph = ops.get_default_graph()
  if not isinstance(obj, six.string_types):
    if not hasattr(obj, "graph") or obj.graph != graph:
      raise ValueError("Passed %s should have graph attribute that is equal "
                       "to current graph %s." % (obj, graph))
    return obj
  if ":" in obj:
    element = graph.as_graph_element(obj)
  else:
    element = graph.as_graph_element(obj + ":0")
    # Check that there is no :1 (e.g. it's single output).
    try:
      graph.as_graph_element(obj + ":1")
    except (KeyError, ValueError):
      pass
    else:
      raise ValueError("Name %s is ambiguous, "
                       "as this `Operation` has multiple outputs "
                       "(at least 2)." % obj)
  return element


class SuperWackHook(session_run_hook.SessionRunHook):
  """Prints the given tensors every N local steps, every N seconds, or at end.
  The tensors will be printed to the log, with `INFO` severity. If you are not
  seeing the logs, you might want to add the following line after your imports:
  ```python
    tf.logging.set_verbosity(tf.logging.INFO)
  ```
  Note that if `at_end` is True, `tensors` should not include any tensor
  whose evaluation produces a side effect such as consuming additional inputs.
  """

  def __init__(self, tensors, every_n_iter=None, every_n_secs=None,
               at_end=False, formatter=None):
    """Initializes a `LoggingTensorHook`.
    Args:
      tensors: `dict` that maps string-valued tags to tensors/tensor names,
          or `iterable` of tensors/tensor names.
      every_n_iter: `int`, print the values of `tensors` once every N local
          steps taken on the current worker.
      every_n_secs: `int` or `float`, print the values of `tensors` once every N
          seconds. Exactly one of `every_n_iter` and `every_n_secs` should be
          provided.
      at_end: `bool` specifying whether to print the values of `tensors` at the
          end of the run.
      formatter: function, takes dict of `tag`->`Tensor` and returns a string.
          If `None` uses default printing all tensors.
    Raises:
      ValueError: if `every_n_iter` is non-positive.
    """
    only_log_at_end = (
        at_end and (every_n_iter is None) and (every_n_secs is None))
    if (not only_log_at_end and
        (every_n_iter is None) == (every_n_secs is None)):
      raise ValueError(
          "either at_end and/or exactly one of every_n_iter and every_n_secs "
          "must be provided.")
    if every_n_iter is not None and every_n_iter <= 0:
      raise ValueError("invalid every_n_iter=%s." % every_n_iter)
    if not isinstance(tensors, dict):
      self._tag_order = tensors
      tensors = {item: item for item in tensors}
    else:
      self._tag_order = sorted(tensors.keys())
    self._tensors = tensors
    self._formatter = formatter
    self._timer = (
        tf.train.NeverTriggerTimer() if only_log_at_end else
        tf.train.SecondOrStepTimer(every_secs=every_n_secs, every_steps=every_n_iter))
    self._log_at_end = at_end

  def begin(self):
    self._timer.reset()
    self._iter_count = 0
    # Convert names to tensors if given
    self._current_tensors = {tag: _as_graph_element(tensor)
                             for (tag, tensor) in self._tensors.items()}

  def before_run(self, run_context):  # pylint: disable=unused-argument
    self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)
    if self._should_trigger:
      return tf.train.SessionRunArgs(self._current_tensors)
    else:
      return None

  def _log_tensors(self, tensor_values):
    original = np.get_printoptions()
    np.set_printoptions(suppress=True)
    elapsed_secs, _ = self._timer.update_last_triggered_step(self._iter_count)
    if self._formatter:
      logging.info(self._formatter(tensor_values))
    else:
      stats = []
      for tag in self._tag_order:
        stats.append("%s = %s" % (tag, tensor_values[tag]))
      if elapsed_secs is not None:
        logging.info("%s (%.3f sec)", ", ".join(stats), elapsed_secs)
      else:
        logging.info("%s", ", ".join(stats))
    np.set_printoptions(**original)

  def after_run(self, run_context, run_values):
    _ = run_context
    if self._should_trigger:
      self._log_tensors(run_values.results)

    self._iter_count += 1

  def end(self, session):
    if self._log_at_end:
      values = session.run(self._current_tensors)
      self._log_tensors(values)

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