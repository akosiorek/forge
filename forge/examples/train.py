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
"""Experiment training script."""
from os import path as osp

import tensorflow as tf

import forge
from forge import flags
import forge.experiment_tools as fet

# job config
flags.DEFINE_string('data_config', 'configs/mnist_data.py', 'Path to a data config file.')
flags.DEFINE_string('model_config', 'configs/mnist_mlp.py', 'Path to a model config file.')
flags.DEFINE_string('results_dir', 'checkpoints', 'Top directory for all experimental results.')
flags.DEFINE_string('run_name', 'mnist', 'Name of this job. Results will be stored in a corresponding folder.')
flags.DEFINE_boolean('resume', False, 'Tries to resume a job if True.')

# logging config
flags.DEFINE_integer('report_loss_every', int(1e3), 'Number of iterations between reporting minibatch loss - hearbeat.')
flags.DEFINE_integer('save_itr', int(1e4), 'Number of iterations between snapshotting the model.')
flags.DEFINE_integer('train_itr', int(1e5), 'Maximum number of training iterations.')

# experiment config
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_float('learning_rate', 1e-5, 'Initial values of the learning rate')

# gpu
flags.DEFINE_string('gpu', '0', 'Id of the gpu to use for this job.')

# Parse flags
config = forge.config()

# sets visible gpus to config.gpu
fet.set_gpu(config.gpu)

# Prepare enviornment
logdir = osp.join(config.results_dir, config.run_name)
logdir, resume_checkpoint = fet.init_checkpoint(logdir, config.data_config, config.model_config, config.resume)
checkpoint_name = osp.join(logdir, 'model.ckpt')

# Build the graph
tf.reset_default_graph()
# load data
data_dict = fet.load(config.data_config, config)
# load the model
loss, stats, _ = fet.load(config.model_config, config, **data_dict)

# Add summaries for reported stats
# summaries can be set up in the model config file
for k, v in stats.iteritems():
    tf.summary.scalar(k, v)

# Print model stats
fet.print_flags()
fet.print_variables_by_scope()
fet.print_num_params()

# Setup the optimizer
global_step = tf.train.get_or_create_global_step()
opt = tf.train.RMSPropOptimizer(config.learning_rate, momentum=.9)

# Create the train step
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = opt.minimize(loss, global_step=global_step)

# create session and initializer variables
sess = fet.get_session()
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
    if train_itr % config.report_loss_every == 0:
        print '{}: {}'.format(train_itr, str(l)[1:-1].replace('\'=', ''))

        if all_summaries is not None:
            summaries = sess.run(all_summaries)
            summary_writer.add_summary(summaries, train_itr)

    if train_itr % config.save_itr == 0:
        saver.save(sess, checkpoint_name, global_step=train_itr)

saver.save(sess, checkpoint_name, global_step=train_itr)
