########################################################################################
# 
# Forge
# Copyright (C) 2018  Adam R. Kosiorek, Oxford Robotics Institute and
#     Department of Statistics, University of Oxford
#
# email:   adamk@robots.ox.ac.uk
# webpage: http://akosiorek.github.io/
# github: https://github.com/akosiorek/forge/
# 
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
# 
########################################################################################

from builtins import range

import numpy as np
import itertools

import tensorflow as tf


def tensors_from_data(data_dict, batch_size, axes=None, shuffle=False):
    """Turns a dict of numpy.ndarrays into a dict of minibatch tensors.

    Arrays are split into minibatches of `batch_size` along `axes`. If `axes` is None,
    then all arrays are split along axis==0. Tensors can iterate sequentially over the
    passed arrays if shuffle=False or in a random order if shuffle=True.

    :param data_dict: dict of {key: nump.ndarray}.
    :param batch_size: integer
    :param axes: dict of {k: integer} or None
    :param shuffle: boolean.
    :return: dict of {key: tf.Tensor}
    """
    keys = list(data_dict.keys())
    if axes is None:
        axes = {k: 0 for k in keys}

    key = keys[0]
    ax = axes[key]
    n_entries = data_dict[key].shape[ax]

    if shuffle:
        def idx_fun():
            return np.random.choice(n_entries, batch_size, replace=False)

    else:
        rolling_idx = itertools.cycle(range(0, n_entries - batch_size + 1, batch_size))

        def idx_fun():
            start = next(rolling_idx)
            end = start + batch_size
            return np.arange(start, end)

    def data_fun():
        idx = idx_fun()
        minibatch = []
        for k in keys:
            item = data_dict[k]
            minibatch_item = item.take(idx, axes[k])
            minibatch.append(minibatch_item)
        return minibatch

    minibatch = data_fun()
    types = [getattr(tf, str(m.dtype)) for m in minibatch]

    tensors = tf.py_func(data_fun, [], types)
    for t, m in zip(tensors, minibatch):
        t.set_shape(m.shape)

    tensors = data_dict.__class__({k: v for k, v in zip(keys, tensors)})
    return tensors