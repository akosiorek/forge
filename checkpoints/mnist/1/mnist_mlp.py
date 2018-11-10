"""Simple MLP model config for MNIST classification."""
import sonnet as snt
import tensorflow as tf

from forge import flags

flags.DEFINE_integer('n_hidden', 128, 'Number of hidden units.')


def load(config, **inputs):

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
