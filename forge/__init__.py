from experiment_tools import load_from_checkpoint, load
from data import tensors_from_data


def config():
    import flags
    return flags.FLAGS