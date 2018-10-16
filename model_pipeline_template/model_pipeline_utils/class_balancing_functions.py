import pdb
import itertools
# Change name to binary onehot classification or something?
def cat_dog_class_balance(dataset):
    cat_pairs = [x for x in dataset if x[1] == [0, 1]]
    dog_pairs = [x for x in dataset if x[1] == [1, 0]]
    print("Length cat pairs: {}, Length dog pairs: {}".format(len(cat_pairs), len(dog_pairs)))
    iters = [iter(cat_pairs), iter(dog_pairs)]
    balanced_pairs = list(it.__next__() for it in itertools.cycle(iters))
    balanced_images, balanced_labels = list(zip(*balanced_pairs))
    return balanced_images, balanced_labels
