import pdb
import sys
import argparse
from model_pipeline_utils.data_pipeline import generate_tfrecords_for_image, imgs_input_fn, get_tfrecords, clear_old_tfrecords, clean_model_dir, create_val_dir, make_json_from_file_names
from model_pipeline_utils.output_tensor_functions import cat_dog_dict_tensor
from model_pipeline_utils.class_balancing_functions import cat_dog_class_balance


def main(train_val_test_split, data_dir, data_format, dict_tensor_string, image_dims, max_records):
    train_frac, val_frac, test_frac = train_val_test_split
    if sum(train_val_test_split) != 1:
        print("ERROR - Train + Val + Test should equal 1")
        return
    dict_tensor_fn, class_balancing_fn = None, None
    if dict_tensor_string == "cat_dog_dict_tensor":
        dict_tensor_fn = cat_dog_dict_tensor
        class_balancing_fn = cat_dog_class_balance
    else:
        print("Function not recognized... exiting")
        return
    clear_old_tfrecords()
    if data_format == "image":
        generate_tfrecords_for_image(data_dir, image_dims, train_frac, val_frac, test_frac, dict_tensor_fn, class_balancing_fn, max_records=max_records)
    else:
        print("Not done...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-val-test-split', type=str, help="Train val test split - should add up to 1.  Should be of form <FLOAT>,<FLOAT>,<FLOAT>")
    parser.add_argument('--data-dir', type=str, help="FULL PATH Location of the data")
    parser.add_argument('--data-format', type=str, help="Data Type: - Images (<LABEL>.jpg, <LABEL>.json). Other types have not yet been implemented")
    parser.add_argument('--dict-tensor-fn', type=str, help="Name of function to turn dicts from JSON files into the required output tensor of the network.")
    parser.add_argument('--image-dims', type=str, help="Image dimensions - should be in the form <HEIGHT>,<WIDTH>,<NUM_CHANNELS>")
    parser.add_argument('--max-records', type=int, default=sys.maxsize, help="Maximum number of records to process")
    args = parser.parse_args()
    # make_json_from_file_names(args.data_dir)
    main([float(item) for item in args.train_val_test_split.split(',')], args.data_dir, args.data_format, args.dict_tensor_fn, [int(item) for item in args.image_dims.split(',')], args.max_records)

# Example Usage: python3 prep_data.py --train-val-test-split 0.8,0.15,0.05 --data-dir '/home/michael/hard_drive/datasets/dogs_vs_cats_data/train/' --data-format image --dict-tensor-fn cat_dog_dict_tensor --image-dims 80,80,3