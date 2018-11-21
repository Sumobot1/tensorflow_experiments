def cnn_model_fn(features, labels, mode, final_dropout_rate, params):
    """Model function for CNN."""
    # NOTE: We are still "in" the model_pipeline_template directory - all imports must be relative to that
    from model_pipeline_utils.tf_models.cat_dog_cnn import cnn_model_fn
    return cnn_model_fn(features, labels, mode, final_dropout_rate, params)


def fast_cnn_model_fn(features, labels, mode, params):
    """Model function for CNN."""
    from tf_models.debugging_cnn import fast_cnn_model_fn
    return fast_cnn_model_fn(features, labels, mode, params)


def yolo_v3():
    from tf_models.yolo_v3 import yolo_v3_model_fn
    return yolo_v3_model_fn()
