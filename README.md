# tensorflow_experiments

An ongoing collection of experiments I am doing to learn more about Tensorflow, and machine learning in general!

![](http://i.imgur.com/ljlUEeC.png)

As with most experiments, this repo is a bit messy.  I will continue to expand/refactor it when I get the chance!

## Contents:
- capsnet_example
	- Jupyter notebook adapted from [oreilly.com](https://www.oreilly.com/ideas/introducing-capsule-networks) on how a capsule network works
	- Based on [this paper](https://arxiv.org/abs/1710.09829)
- cnn_mnist_example
	- Modified tutorial from the [Tensorflow docs](https://www.tensorflow.org/tutorials/estimators/cnn) used to look into how to use Tensorflow's Estimators.
- model_pipeline_template
	- A model training pipeline created using Tensorflow.  Contains multiple files/folders
		- prep_data.py
			- Loads and preprocesses image data, writes it to train/valid/test tfrecord files
			- TODO: Make tfrecord generation work for non-image data as well
		- train_model.py
			- Trains a model from scratch, or from a checkpoint
			- Uses tf.data's iterators to load in training/validation data
			- Saves highest performing validation checkpoints, as well as predictions/results for later analysis
			- I think it works with Tensorboard, but have not really tested that yet
		- export_model.py
			- Exports specified model as a frozen graph, which can be loaded to make predictions
		- make_predictions.py
			- Loads the frozen graph, to make predictions on a directory of images.
		- model_pipeline_utils
			- A collection of helper functions, etc. that make the model training pipeline work
		- validation_analysis
			- Jupyter notebooks to analyze training and validation images/results
	- Currently using the Model Training Pipeline as a base to work on implementing the YOLOv3 object detection network based on [this paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
- TODO: kaggle_problems
	- After I have cleaned up/organized them, I would like to add my dog/cat and toxic comment classifiers
