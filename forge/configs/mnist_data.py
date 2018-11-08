from attrdict import AttrDict
from tensorflow.examples.tutorials.mnist import input_data

from forge import tf_flags as flags
from forge.data import tensors_from_data


flags.DEFINE_string('data_folder', '.', 'Path to a data folder.')


def load(batch_size, **kwargs):

    del kwargs
    F = flags.FLAGS

    dataset = input_data.read_data_sets(F.data_folder)

    train_data = {'imgs': dataset.train.images, 'labels': dataset.train.labels}
    valid_data = {'imgs': dataset.validation.images, 'labels': dataset.validation.labels}

    train_tensors = tensors_from_data(train_data, batch_size, shuffle=True)
    valid_tensors = tensors_from_data(valid_data, batch_size, shuffle=False)

    data_dict = AttrDict(
        train_img=train_tensors['imgs'],
        valid_img=valid_tensors['imgs'],
        train_label=train_tensors['labels'],
        valid_label=valid_tensors['labels'],
    )

    return data_dict


