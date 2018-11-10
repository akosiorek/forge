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
