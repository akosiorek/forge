###############################################################################
#
# Forge
# Copyright (C) 2019    Martin Engelcke,
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


import os

import torch
from torchvision import datasets, transforms

from forge import flags


flags.DEFINE_string('data_folder', 'data/MNIST_data', 'Path to data folder.')


def load(config, **unused_kwargs):
    """
    Loads a dataset.

    Args:
        config: (dict): write your description
        unused_kwargs: (dict): write your description
    """
    del unused_kwargs

    if not os.path.exists(config.data_folder):
        os.makedirs(config.data_folder)

    mnist = datasets.MNIST(config.data_folder, train=True, download=True,
                           transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(
        mnist, batch_size=config.batch_size, shuffle=True)

    return train_loader
