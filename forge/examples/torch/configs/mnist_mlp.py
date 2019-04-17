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


import torch
import torch.nn as nn

from forge import flags


flags.DEFINE_integer('n_hidden', 128, 'Number of hidden units.')


def load(config):
    mlp = nn.Sequential(
        nn.Linear(784, config.n_hidden),
        nn.ReLU(),
        nn.Linear(config.n_hidden, 10),
    )
    return mlp
