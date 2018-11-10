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

"""Example script that loads a saved model from a checkpoint."""
import tensorflow as tf
from forge import load_from_checkpoint


checkpoint_dir = '../checkpoints/mnist/1'
checkpoint_iter = int(1e4)

data, model_parts, restore_func = load_from_checkpoint(checkpoint_dir, checkpoint_iter)
artefacts = model_parts[-1]

sess = tf.Session()
restore_func(sess)

print 'Loaded artefacts:'
for k, v in artefacts.iteritems():
    print k, v

# Model is now loaded and ready to use.
