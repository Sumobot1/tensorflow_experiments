import tensorflow as tf
from tensorflow.python.training import session_run_hook
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.framework import ops
import six
import numpy as np
from tensorflow.python.platform import tf_logging as logging
import pdb
from termcolor import colored, cprint


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


# Copied from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/basic_session_run_hooks.py
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
                 at_end=False, formatter=None, total_num_steps=None):
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
        if (not only_log_at_end and (every_n_iter is None) == (every_n_secs is None)):
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
        self._total_num_steps = total_num_steps

    def begin(self):
        self._timer.reset()
        self._iter_count = 0
        # Convert names to tensors if given
        self._current_tensors = {tag: _as_graph_element(tensor) for (tag, tensor) in self._tensors.items()}

    def before_run(self, run_context):  # pylint: disable=unused-argument
        self._should_trigger = self._timer.should_trigger_for_step(self._iter_count) or self._total_num_steps - 1 <= self._iter_count
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
                if tag is not 'batch_size':
                    stats.append('{}'.format("{} = {:7.4f}".format(tag, tensor_values[tag])))
            stats = ['Step: {:5d}/{}'.format(self._iter_count, self._total_num_steps - 1)] + stats
            cprint("{}".format(', '.join(stats)), 'green', end='\n' if (self._total_num_steps - self._iter_count - tensor_values['batch_size'] <= 0) else '\r')
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
