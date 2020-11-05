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

"""Simple MLP model config for MNIST classification."""
import sonnet as snt
import tensorflow as tf

from forge import flags

flags.DEFINE_integer('n_hidden', 128, 'Number of hidden units.')


def load(config, **inputs):
    """
    Load the model.

    Args:
        config: (dict): write your description
        inputs: (str): write your description
    """

    imgs, labels = inputs['train_img'], inputs['train_label']

    imgs = snt.BatchFlatten()(imgs)
    mlp = snt.nets.MLP([config.n_hidden, 10])
    logits = mlp(imgs)
    labels = tf.cast(labels, tf.int32)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    pred_class = tf.argmax(logits, -1)
    acc = tf.reduce_mean(tf.to_float(tf.equal(tf.to_int32(pred_class), labels)))

    # put here everything that you might want to use later
    # for example when you load the model in a jupyter notebook
    artefects = {
        'mlp': mlp,
        'logits': logits,
        'loss': loss,
        'pred_class': pred_class,
        'accuracy': acc
    }

    # put here everything that you'd like to be reported every N training iterations
    # as tensorboard logs AND on the command line
    stats = {'crossentropy': loss, 'accuracy': acc}

    # loss will be minimized with respect to the model parameters
    return loss, stats, artefects
