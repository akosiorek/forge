"""Experiment script."""
import os
from os import path as osp

import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')

from forge.experiment_tools import (load, init_checkpoint, parse_flags, get_session, print_flags,
                                    print_num_params, print_variables_by_scope)
from forge import tf_flags as flags

# Define flags

flags.DEFINE_string('data_config', 'configs/mnist_data.py', 'Path to a data config file.')
flags.DEFINE_string('model_config', 'configs/mnist_mlp.py', 'Path to a model config file.')
flags.DEFINE_string('results_dir', '../checkpoints', 'Top directory for all experimental results.')
flags.DEFINE_string('run_name', 'test_run', 'Name of this job. Results will be stored in a corresponding folder.')

flags.DEFINE_integer('batch_size', 32, '')

flags.DEFINE_integer('log_itr', int(1e4), 'Number of iterations between storing tensorboard logs.')
flags.DEFINE_integer('report_loss_every', int(1e3), 'Number of iterations between reporting minibatch loss - hearbeat.')
flags.DEFINE_integer('save_itr', int(1e5), 'Number of iterations between snapshotting the model.')
flags.DEFINE_integer('fig_itr', 10000, 'Number of iterations between creating results figures.')
flags.DEFINE_integer('train_itr', int(2e6), 'Maximum number of training iterations.')

flags.DEFINE_boolean('log_at_start', False, 'Evaluates the model between training commences if True.')
flags.DEFINE_boolean('eval_on_train', True, 'Evaluates the model on the train set if True')

flags.DEFINE_float('eval_size_fraction', 1., 'Fraction of the dataset to perform model evaluation on. Must be between'
                                             '0. and 1.')

flags.DEFINE_string('opt', 'rmsprop', 'Optimizer; choose from rmsprop, adam, sgd, momentum')
flags.DEFINE_float('learning_rate', 1e-5, 'Initial values of the learning rate')
flags.DEFINE_string('schedule', '4,6,10', 'Uses a learning rate schedule if True. Schedule = \'4,6,10\' '
                                           'means that F.train_itr will be split in proportions 4/s, 6/s, 10/s,'
                                           'where s = sum(schedule)')

flags.DEFINE_string('gpu', '0', 'Id of the gpu to use for this job.')
flags.DEFINE_boolean('debug', False, 'Adds a lot of tensorboard summaries if True.')

F = flags.FLAGS
os.environ['CUDA_VISIBLE_DEVICES'] = F.gpu

# Parse flags
parse_flags()
F = flags.FLAGS

# Prepare enviornment
logdir = osp.join(F.results_dir, F.run_name)
logdir, flags, resume_checkpoint = init_checkpoint(logdir, F.data_config, F.model_config, F.resume)
checkpoint_name = osp.join(logdir, 'model.ckpt')

# Build the graph
tf.reset_default_graph()
data_dict = load(F.data_config, F.batch_size)

model_kwargs = dict(**data_dict)
model_kwargs['debug'] = F.debug
loss, stats, plots = load(F.model_config, **model_kwargs)

# Print model stats
print_flags()
print_variables_by_scope()
print_num_params()

# Setup the optimizer
global_step = tf.train.get_or_create_global_step()
lr = F.learning_rate
if F.schedule:
    schedule = [float(f) for f in F.schedule.split(',')]
    schedule = np.cumsum(schedule)
    schedule = schedule * F.train_itr / schedule[-1]
    schedule = list(np.round(schedule).astype(np.int32))
    lrs = list(lr * (1./3) ** np.arange(len(schedule)))
    print lrs, schedule
    lr = tf.train.piecewise_constant(global_step, schedule[:-1], lrs)
    tf.summary.scalar('learning_rate', lr)

opt = F.opt.lower()
if opt == 'rmsprop':
    opt = tf.train.RMSPropOptimizer(lr, momentum=.9)
elif opt == 'adam':
    opt = tf.train.AdamOptimizer(lr)
elif opt == 'sgd':
    opt = tf.train.GradientDescentOptimizer(lr)
elif opt == 'momentum':
    opt = tf.train.MomentumOptimizer(lr, momentum=.9)


# Optimisation target
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = opt.minimize(loss, global_step=global_step)

sess = get_session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(max_to_keep=10000)
if resume_checkpoint is not None:
    print "Restoring checkpoint from '{}'".format(resume_checkpoint)
    saver.restore(sess, resume_checkpoint)

summary_writer = tf.summary.FileWriter(logdir, sess.graph)
all_summaries = tf.summary.merge_all()

# Logging
# TODO: logging in some general way

train_itr = sess.run(global_step)
print 'Starting training at iter = {}'.format(train_itr)

# if F.log_at_start or train_itr == 0:
#    log(train_itr)
#    try_plot(train_itr)

# Train!
while train_itr < F.train_itr:
    l, train_itr, _ = sess.run([stats, global_step, train_step])

    if train_itr % F.report_loss_every == 0:
        print '{}: {}'.format(train_itr, str(l)[1:-1].replace('\'=', ''))
        summaries = sess.run(all_summaries)
        summary_writer.add_summary(summaries, train_itr)

    # if train_itr != 0 and train_itr % F.log_itr == 0:
    #     log(train_itr)

    if train_itr % F.save_itr == 0:
        saver.save(sess, checkpoint_name, global_step=train_itr)

    # if train_itr % F.fig_itr == 0:
    #     try_plot(train_itr)

saver.save(sess, checkpoint_name, global_step=train_itr)
# try_plot(train_itr)
