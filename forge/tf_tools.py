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
from __future__ import print_function

import imp
import importlib
import os
import os.path as osp
import sys
import re
import shutil
import simplejson as json
import subprocess

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np

from forge import flags as _flags
from experiment_tools import format_integer, json_load


FLAG_FILE = 'flags.json'


def load_from_checkpoint(checkpoint_dir, checkpoint_iter, path_prefix='',
                         model_kwargs=None, data_kwargs=None, override_flags=None):
    """Loads model and data from a specified checkpoint.

    An example would be:
    >>> dir = '../checkpoints/vae/1'
    >>> iter = int(1e5)
    >>> data, model, restore = load_from_checkpoint(dir, iter)
    >>> sess = tf.Session()
    >>> restore(sess) # a this point model parameters are restored

    :param checkpoint_dir: Checkpoint directory containing model checkpoints and the flags.json file.
    :param checkpoint_iter: int, global-step of the checkpoint to be loaded.
    :param path_prefix: string; path to be appended to config paths in case they were saved as non-absolute paths.
    :param model_kwargs: dict of kwargs passed to the model loading in addition to data.
    :param data_kwargs: dict of kwargs passed to data loading.
    :param override_flags: dict of kwargs used to override values restored from the flag file.
    :return: (data, model, restore_func), where data and model are loaded from their corresponding config files.
        Calling `restore_func(sess)`, which takes a tf.Session as an argument, restores model parameters.
    """
    flags = json_load(osp.join(checkpoint_dir, FLAG_FILE))
    if override_flags is not None:
        flags.update(override_flags)

    _restore_flags(flags)
    F = _flags.FLAGS

    # Load data and model and figure out which trainable variables should be loaded with the model.
    all_train_vars_before = set(tf.trainable_variables())
    # TODO(akosiorek): this should use config files stored in the job folder, not the ones
    # that the config file is pointing to.

    data_kwargs = data_kwargs if data_kwargs is not None else {}
    data = load(path_prefix + F.data_config, F, **data_kwargs)

    if model_kwargs is not None:
        data.update(model_kwargs)

    model = load(path_prefix + F.model_config, F, **data)
    all_train_vars_after = set(tf.trainable_variables())
    model_vars = list(all_train_vars_after - all_train_vars_before)

    checkpoint_path = osp.join(checkpoint_dir, 'model.ckpt-{}'.format(checkpoint_iter))

    def restore_func(sess):
        print('Restoring model from "{}"'.format(checkpoint_path))
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(model_vars)
        saver.restore(sess, checkpoint_path)

    return data, model, restore_func


def print_num_params():
    num_params = sum([np.prod(v.shape.as_list(), dtype=int) for v in tf.trainable_variables()])
    num_params = format_integer(num_params)
    print('Number of trainable parameters: {}'.format(num_params))


def print_variables_by_scope():
    """Prints trainable variables by scope."""
    vars = [(v.name, v.shape.as_list()) for v in tf.trainable_variables()]
    vars = sorted(vars, key=lambda x: x[0])

    last_scope = None
    scope_n_params = 0
    for i, (name, shape) in enumerate(vars):

        current_scope = name.split('/', 1)[0]
        if current_scope != last_scope:
            if last_scope is not None:
                scope_n_params = format_integer(scope_n_params)
                print('\t#  scope params = {}'.format(scope_n_params))
                print

            print('scope:', current_scope)
            scope_n_params = 0

        last_scope = current_scope
        n_params = np.prod(shape, dtype=np.int32)
        scope_n_params += n_params
        print('\t', name, shape)

    print('\t#  scope params = {}'.format(format_integer(scope_n_params)))
    print


def get_session(tfdbg=False):
    """Utility function for getting tf.Session."""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    if tfdbg:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    return sess
