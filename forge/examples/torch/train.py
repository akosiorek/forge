###############################################################################
#
# Forge
# Copyright (C) 2018    Martin Engelcke,
#                       Applied Artificial Intelligence Lab,
#                       Oxford Robotics Institute,
#                       University of Oxford
#
# email: martin@robots.ox.ac.uk
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
###############################################################################


from __future__ import print_function
from os import path as osp

import torch
import torch.nn.functional as F
import torch.optim as optim

import forge
from forge import flags
import forge.experiment_tools as fet


# Job config
flags.DEFINE_string('data_config', 'configs/mnist_data.py',
                    'Path to a data config file.')
flags.DEFINE_string('model_config', 'configs/mnist_mlp.py',
                    'Path to a model config file.')
flags.DEFINE_string('results_dir', 'checkpoints',
                    'Top directory for all experimental results.')
flags.DEFINE_string('run_name', 'mnist',
                    'Name of this job and name of results folder.')
flags.DEFINE_boolean('resume', False, 'Tries to resume a job if True.')

# Logging config
flags.DEFINE_integer('report_loss_every', 100,
                     'Number of iterations between reporting minibatch loss.')
flags.DEFINE_integer('train_epochs', 20, 'Maximum number of training epochs.')

# Experiment config
flags.DEFINE_integer('batch_size', 32, 'Mini-batch size.')
flags.DEFINE_float('learning_rate', 1e-5, 'SGD learning rate.')

# Parse flags
config = forge.config()

# Prepare enviornment
logdir = osp.join(config.results_dir, config.run_name)
logdir, resume_checkpoint = fet.init_checkpoint(
    logdir, config.data_config, config.model_config, config.resume)
checkpoint_name = osp.join(logdir, 'model.ckpt')

# Load data
train_loader = fet.load(config.data_config, config)
# Load model
model = fet.load(config.model_config, config)

# Print flags
fet.print_flags()
# Print model info
print(model)

# Setup optimizer
optimizer = optim.RMSprop(model.parameters(),
                          lr=config.learning_rate,
                          momentum=0.9)

# Try to restore model and optimizer from checkpoint
if resume_checkpoint is not None:
    print("Restoring checkpoint from '{}'".format(resume_checkpoint))
    checkpoint = torch.load(resume_checkpoint)
    # Restore model
    model.load_state_dict(checkpoint['model_state_dict'])
    # Restore optimizer
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # Update starting epoch
    start_epoch = checkpoint['epoch'] + 1
else:
    start_epoch = 1
print("Starting training at iter = {}".format(start_epoch))

# Training
for epoch in range(start_epoch, config.train_epochs + 1):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data.view(-1, 784))
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        # Logging
        if batch_idx % config.report_loss_every == 0:
            pred = output.max(1)[1]
            correct = pred.eq(target.view_as(pred)).sum().item()
            l = ['crossentropy:', loss.item(), 'accuracy:',
                 float(correct) / len(data)]
            print('epoch: {} [{} / {}]: {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), l))

            # TODO(martinengelcke): add support for Tensorboard logging?
            # e.g. via https://github.com/lanpa/tensorboardX

    epoch_ckpt_file = '{}-{}'.format(checkpoint_name, epoch)
    print("Saving model training checkpoint to {}".format(epoch_ckpt_file))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, epoch_ckpt_file)
