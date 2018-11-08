"""Sample model config.
"""
import sonnet as snt
import tensorflow as tf

from forge import tf_flags as flags

flags.DEFINE_integer('n_hidden', 128, 'Number of hidden units.')


def load(**inputs):
    F = flags.FLAGS

    imgs, labels = inputs['train_img'], inputs['train_label']

    imgs = snt.BatchFlatten()(imgs)
    logits = snt.nets.MLP([F.n_hidden, 10])(imgs)
    labels = tf.cast(labels, tf.int32)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    stats = {'xe': loss}
    plots = None

    return loss, stats, plots
