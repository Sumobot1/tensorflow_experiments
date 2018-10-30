import pandas as pd
from termcolor import cprint
from model_pipeline_utils.train_model_utils import show_image


def interactive_check_predictions(img, prediction):
    print("cat" if prediction[0] == 1 else "dog")
    show_image(img)


def cat_dog_classifier_output(test_predictions, test_ids, output_labels):
    test_ids, test_preds = list(zip(*sorted(list(zip(test_ids, test_predictions)))))
    df = {output_labels[0]: test_ids, output_labels[1]: [arr[0] for arr in test_preds[0]]}
    df = pd.DataFrame(data=df)
    df.to_csv("predictions.csv", sep=',', index=False)


def get_appropriate_prediction_fn(output_func):
    if output_func == "cat_dog_classifier_output":
        return cat_dog_classifier_output
    else:
        cprint("Prediction function is not made yet - go make it?", "red")