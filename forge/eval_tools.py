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

"""Tools used for model evaluation."""
import collections
import time

import tensorflow as tf
from tensorflow.python.util import nest


def make_expr_logger(sess, num_batches, expr_dict, name, data_dict=None,
                     constants_dict=None, measure_time=True, writer=None):
    """Creates a logging function which evaluates expressions in `expr_dict` when called.

    The logger evaluates expressions from `expr_dict` and takes avereges over `num_batches`.
    Expressions are evaluated as-is if `data_dict` is None. Otherwise, `data_dict` should contain
    {target_tensor: source_tensor} pairs, where data from the source_tensor is passed as a
    target_tensor into the graph with the feed dict mechanism.

    Logs are written to standard output and stored as Tensorboard summaries if `writer` is not None.

    An example:
    >>> # assume `imgs`, `labels` are given and represent trainset tensors
    >>> logits = create_model(imgs)
    >>> acc = compute_accuraccy(logits, labels)
    >>> expressions = {'accuracy': acc, 'mean_logit': tf.reduce_mean(logits)}
    >>> # we would like to evaluate it on the training set
    >>> data_dict = {imgs: test_imgs, labels: test_labels}  # test_imgs and test_lables are tensors
    >>> # when run in a session, test_(imgs/labels) produce numpy.ndarrays
    >>>
    >>> logdir = '../checkpoints/run/1/'
    >>> writer = tf.summary.filewriter(logdir)
    >>> logger = make_expr_logger(sess, num_batches=10, expr_dict=expressions, data_dict=data_dict, writer=writer)
    >>>
    >>> # ... train the model
    >>> logger(train_iter)  # evaluates expressions on 10 minibatches of test data and stores tensorboard logs
    >>> # as well as prints the training iterations, values of the expressions and time taken to evalaute to
    >>> # standard error

    :param sess: tf.Session used to evaluate expressions and data tensors.
    :param num_batches: Integer, number of minibatches to use for evaluation.
    :param expr_dict: dict of {string: tf.Tensor}, expressions to be evaluated.
    :param name: string, name appended to expressions' names.
    :param data_dict: dict of {target_tensor: source_tensor} containing data feeds.
    :param constants_dict: dict of {target_tensor: constant} that is added to the feed dict.
    :param measure_time: boolean, if True, time taken to evaluate the expressions is reported.
    :param writer: tf.FileWriter object. If present, logs are stored as Tensorboard summaries.
    :return: callable, takes training iteration as input.
    """

    expr_dict = collections.OrderedDict(sorted(expr_dict.items()))

    tags = {k: '/'.join((k, name)) for k in expr_dict}
    data_name = 'Data {}'.format(name)
    log_string = ', '.join((''.join((k + ' = {', k, ':.4f}')) for k in expr_dict))
    log_string = ' '.join(('Step {},', data_name, log_string))

    if measure_time:
        log_string += ', eval time = {:.4}s'

        def make_log_string(itr, l, t):
            return log_string.format(itr, t, **l)
    else:
        def make_log_string(itr, l, t):
            return log_string.format(itr, **l)

    def log(itr, l, t):
        try:
            return make_log_string(itr, l, t)
        except ValueError as err:
            print err.message
            print '\tLogging items'
            for k, v in l.iteritems():
                print '{}: {}'.format(k, type(v))

    def logger(itr=0, num_batches_to_eval=None, write=True, writer=writer):
        l = {k: 0. for k in expr_dict}
        start = time.time()
        if num_batches_to_eval is None:
            num_batches_to_eval = num_batches

        for i in xrange(num_batches_to_eval):
            if data_dict is not None:
                vals = sess.run(data_dict.values())
                feed_dict = {k: v for k, v in zip(data_dict.keys(), vals)}
                if constants_dict:
                    feed_dict.update(constants_dict)
            else:
                feed_dict = constants_dict

            r = sess.run(expr_dict, feed_dict)
            for k, v in r.iteritems():
                l[k] += v

        for k, v in l.iteritems():
            l[k] /= num_batches_to_eval
        t = time.time() - start
        print log(itr, l, t)

        if writer is not None and write:
            log_values(writer, itr, [tags[k] for k in l.keys()], l.values())

        return l

    return logger


def log_ratio(var_tuple, name='ratio', eps=1e-8):
    """Creates a scalar summary of the ratio of tensors in `var_tuple`.

    :param var_tuple: tuple of Tensors
    :param name: string.
    :param eps: float.
    """
    a, b = var_tuple
    ratio = tf.reduce_mean(abs(a) / (abs(b) + eps))
    tf.summary.scalar(name, ratio)


def log_norm(expr_list, name):
    """Creates a scalar summary of the norm of the list of tensors.

    :param expr_list: tensor or a list of Tensors.
    :param name: string
    """
    n_elems = 0
    norm = 0.
    for e in nest.flatten(expr_list):
        n_elems += tf.reduce_prod(tf.shape(e))
        norm += tf.reduce_sum(e ** 2)
    norm /= tf.to_float(n_elems)
    tf.summary.scalar(name, norm)
    return norm


def log_values(writer, itr, tags=None, values=None, dict=None):
    """Writes scalar summaries to Tensorboard.

    Values can be passed as either a list of string tags and float values, or
    as a dictionary. In the latter case, the keys are used as summary tags.

    :param writer: tf.summary.Filewriter
    :param itr: int, training iteration
    :param tags: list of strings or None
    :param values: list of floats or None
    :param dict: dict of {string: float} or None
    """
    if dict is not None:
        assert tags is None and values is None
        tags = dict.keys()
        values = dict.values()
    else:

        if not nest.is_sequence(tags):
            tags, values = [tags], [values]

        elif len(tags) != len(values):
            raise ValueError('tag and value have different lenghts:'
                             ' {} vs {}'.format(len(tags), len(values)))

    for t, v in zip(tags, values):
        summary = tf.Summary.Value(tag=t, simple_value=v)
        summary = tf.Summary(value=[summary])
        writer.add_summary(summary, itr)


def gradient_summaries(gvs, norm=True, ratio=True, histogram=True):
    """Register gradient summaries.

    Logs the global norm of the gradient, ratios of gradient_norm/uariable_norm and
    histograms of gradients.

    :param gvs: list of (gradient, variable) tuples
    :param norm: boolean, logs norm of the gradient if True
    :param ratio: boolean, logs ratios if True
    :param histogram: boolean, logs gradient histograms if True
    """

    with tf.name_scope('grad_summary'):
        if norm:
            grad_norm = tf.global_norm([gv[0] for gv in gvs])
            tf.summary.scalar('grad_norm', grad_norm)

        for g, v in gvs:
            var_name = v.name.split(':')[0]
            if g is None:
                print 'Gradient for variable {} is None'.format(var_name)
                continue

            if ratio:
                log_ratio((g, v), '/'.join(('grad_ratio', var_name)))

            if histogram:
                tf.summary.histogram('/'.join(('grad_hist', var_name)), g)
