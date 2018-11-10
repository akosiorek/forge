"""Experiment training script."""
from os import path as osp

import tensorflow as tf

from forge.experiment_tools import (load, init_checkpoint, parse_flags, get_session, print_flags,
                                    print_num_params, print_variables_by_scope, set_gpu)
from forge import flags as flags

# job config
flags.DEFINE_string('data_config', 'configs/mnist_data.py', 'Path to a data config file.')
flags.DEFINE_string('model_config', 'configs/mnist_mlp.py', 'Path to a model config file.')
flags.DEFINE_string('results_dir', '../checkpoints', 'Top directory for all experimental results.')
flags.DEFINE_string('run_name', 'test_run', 'Name of this job. Results will be stored in a corresponding folder.')
flags.DEFINE_boolean('resume', False, 'Tries to resume a job if True.')

# logging config
flags.DEFINE_integer('report_loss_every', int(1e3), 'Number of iterations between reporting minibatch loss - hearbeat.')
flags.DEFINE_integer('save_itr', int(1e4), 'Number of iterations between snapshotting the model.')
flags.DEFINE_integer('train_itr', int(2e6), 'Maximum number of training iterations.')

# experiment config
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_float('learning_rate', 1e-5, 'Initial values of the learning rate')

# gpu
flags.DEFINE_string('gpu', '0', 'Id of the gpu to use for this job.')

F = flags.FLAGS

# sets visible gpus to F.gpu
set_gpu(F.gpu)

# Parse flags
parse_flags()
config = flags.FLAGS

# Prepare enviornment
logdir = osp.join(config.results_dir, config.run_name)
logdir, flags, resume_checkpoint = init_checkpoint(logdir, config.data_config, config.model_config, config.resume)
checkpoint_name = osp.join(logdir, 'model.ckpt')

# Build the graph
tf.reset_default_graph()
# load data
data_dict = load(config.data_config, config)
# load the model
loss, stats, _ = load(config.model_config, config, **data_dict)

# Add summaries for reported stats
# summaries can be set up int he model config file
for k, v in stats.iteritems():
    tf.summary.scalar(k, v)

# Print model stats
print_flags()
print_variables_by_scope()
print_num_params()

# Setup the optimizer
global_step = tf.train.get_or_create_global_step()
opt = tf.train.RMSPropOptimizer(config.learning_rate, momentum=.9)

# Create the train step
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = opt.minimize(loss, global_step=global_step)

# create session and initializer variables
sess = get_session()
sess.run(tf.global_variables_initializer())

# Try to restore the model from a checkpoint
saver = tf.train.Saver(max_to_keep=10000)
if resume_checkpoint is not None:
    print "Restoring checkpoint from '{}'".format(resume_checkpoint)
    saver.restore(sess, resume_checkpoint)

summary_writer = tf.summary.FileWriter(logdir, sess.graph)
all_summaries = tf.summary.merge_all()

train_itr = sess.run(global_step)
print 'Starting training at iter = {}'.format(train_itr)

# Train!
while train_itr < config.train_itr:
    l, train_itr, _ = sess.run([stats, global_step, train_step])

    # tensorboard summaries and heartbeat logs
    if train_itr % F.report_loss_every == 0:
        print '{}: {}'.format(train_itr, str(l)[1:-1].replace('\'=', ''))

        if all_summaries is not None:
            summaries = sess.run(all_summaries)
            summary_writer.add_summary(summaries, train_itr)

    if train_itr % F.save_itr == 0:
        saver.save(sess, checkpoint_name, global_step=train_itr)

saver.save(sess, checkpoint_name, global_step=train_itr)
