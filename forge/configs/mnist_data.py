"""MNIST data config."""
from attrdict import AttrDict
import os
from tensorflow.examples.tutorials.mnist import input_data

from forge import flags
from forge.data import tensors_from_data


flags.DEFINE_string('data_folder', '../data/MNIST_data', 'Path to a data folder.')


# This function should return a dataset in a form that is accepted by the
# corresponding model file.
# In this case, it returns a dictionary of tensors.
def load(config, **unused_kwargs):

    del unused_kwargs

    if not os.path.exists(config.data_folder):
        os.makedirs(config.data_folder)

    dataset = input_data.read_data_sets(config.data_folder)

    train_data = {'imgs': dataset.train.images, 'labels': dataset.train.labels}
    valid_data = {'imgs': dataset.validation.images, 'labels': dataset.validation.labels}

    # This function turns a dictionary of numpy.ndarrays into tensors.
    train_tensors = tensors_from_data(train_data, config.batch_size, shuffle=True)
    valid_tensors = tensors_from_data(valid_data, config.batch_size, shuffle=False)

    data_dict = AttrDict(
        train_img=train_tensors['imgs'],
        valid_img=valid_tensors['imgs'],
        train_label=train_tensors['labels'],
        valid_label=valid_tensors['labels'],
    )

    return data_dict


