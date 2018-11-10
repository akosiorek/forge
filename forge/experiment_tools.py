"""Tools used by the experiment script.
"""
import imp
import importlib
import os
import os.path as osp
import sys
import re
import shutil
import json
import subprocess

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

sys.path.append('../')
from forge import tf_flags

FLAG_FILE = 'flags.json'

# TODO: docs


def json_store(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, sort_keys=True, indent=4)


def json_load(path):
    with open(path, 'r') as f:
        return json.load(f)


def load_from_checkpoint(checkpoint_dir, checkpoint_iter, path_prefix=''):
    """Loads model and data from a specified checkpoint.

    An example would be:
    >>> dir = '../checkpoints/vae/1'
    >>> iter = int(1e5)
    >>> data, model, restore = load_from_checkpoint(dir, iter)
    >>> sess = tf.Session()
    >>> model.load(sess) # a this point model parameters are restored

    :param checkpoint_dir: Checkpoint directory containing model checkpoints and the flags.json file.
    :param checkpoint_iter: int, global-step of the checkpoint to be loaded.
    :param path_prefix: string; path to be appended to config paths in case they were saved as non-absolute paths.
    :return: (data, model), where data and model are loaded from their corresponding config files.
        The model has a `load` function, which takes a tf.Session as an argument and restores model parameters.
    """
    flags = json_load(osp.join(checkpoint_dir, 'flags.json'))
    _restore_flags(flags)
    F = tf_flags.FLAGS

    # Load data and model and figure out which trainable variables should be loaded with the model.
    all_train_vars_before = set(tf.trainable_variables())
    data = load(path_prefix + F.data_config, F.batch_size)
    model = load(path_prefix + F.model_config, **data)
    all_train_vars_after = set(tf.trainable_variables())
    model_vars = list(all_train_vars_after - all_train_vars_before)

    checkpoint_path = osp.join(checkpoint_dir, 'model.ckpt-{}'.format(checkpoint_iter))

    def restore_func(sess):
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(model_vars)
        saver.restore(sess, checkpoint_path)

    model.load = restore_func

    return data, model


def init_checkpoint(checkpoint_dir, data_config, model_config, resume):
    """
    1) try mk checkpoint_dir
    2) if continue:
        a) check if checkpoint_dir/n where n is an integer and raise if it doesnt
        b) load flags
        c) load checkpoint

    3) if not:
        a) n=n+1, mkdir
        b) store flags
        c) copy data & model configs

    :param checkpoint_dir:
    :param data_config:
    :param model_config:
    :return:
    """

    # Make sure these are absolute paths as otherwise model loading becomes tricky.
    data_config, model_config = (osp.abspath(i) for i in (data_config, model_config))

    # check if the experiment folder exists and create if not
    checkpoint_dir_exists = os.path.exists(checkpoint_dir)
    if not checkpoint_dir_exists:
        if resume:
            raise ValueError("Can't resume when the checkpoint dir '{}' doesn't exist.".format(checkpoint_dir))
        else:
            os.makedirs(checkpoint_dir)

    elif not os.path.isdir(checkpoint_dir):
        raise ValueError("Checkpoint dir '{}' is not a directory.".format(checkpoint_dir))

    experiment_folders = [f for f in os.listdir(checkpoint_dir)
                          if not f.startswith('_') and not f.startswith('.')]
    
    if experiment_folders:
        experiment_folder = int(sorted(experiment_folders, key=lambda x: int(x))[-1])
        if not resume:
            experiment_folder += 1
    else:
        if resume:
            raise ValueError("Can't resume since no experiments were run before in checkpoint"
                             " dir '{}'.".format(checkpoint_dir))
        else:
            experiment_folder = 1

    experiment_folder = os.path.join(checkpoint_dir, str(experiment_folder))
    if not resume:
        os.mkdir(experiment_folder)

    flag_path = os.path.join(experiment_folder, FLAG_FILE)
    resume_checkpoint = None

    _load_flags(model_config, data_config)
    flags = parse_flags()
    assert_all_flags_parsed()

    if resume:
        restored_flags = json_load(flag_path)
        flags.update(restored_flags)
        _restore_flags(flags)
        model_files = find_model_files(experiment_folder)
        if model_files:
            resume_checkpoint = model_files[max(model_files.keys())]

    else:
        # store flags
        try:
            flags['git_commit'] = get_git_revision_hash()
        except subprocess.CalledProcessError:
            # not in repo
            pass

        json_store(flag_path, flags)

        # store configs
        for src in (model_config, data_config):
            file_name = os.path.basename(src)
            dst = os.path.join(experiment_folder, file_name)
            shutil.copy(src, dst)

    return experiment_folder, flags, resume_checkpoint


def extract_itr_from_modelfile(model_path):
    return int(model_path.split('-')[-1].split('.')[0])


def find_model_files(model_dir):
    pattern = re.compile(r'.ckpt-[0-9]+$')
    model_files = [f.replace('.index', '') for f in os.listdir(model_dir)]
    model_files = [f for f in model_files if pattern.search(f)]
    model_files = {extract_itr_from_modelfile(f): os.path.join(model_dir, f) for f in model_files}
    return model_files


def load(conf_path, *args, **kwargs):

    module, name = _import_module(conf_path)
    try:
        load_func = module.load
    except AttributeError:
        raise ValueError("The config file should specify 'load' function but no such function was "
                           "found in {}".format(module.__file__))

    print "Loading '{}' from {}".format(module.__name__, module.__file__)
    return load_func(*args, **kwargs)


def _import_module(module_path_or_name):
    module, name = None, None

    if module_path_or_name.endswith('.py'):

        if not os.path.exists(module_path_or_name):
            raise RuntimeError('File {} does not exist.'.format(module_path_or_name))

        file_name = module_path_or_name
        module_path_or_name = os.path.basename(os.path.splitext(module_path_or_name)[0])
        if module_path_or_name in sys.modules:
            module = sys.modules[module_path_or_name]
        else:
            module = imp.load_source(module_path_or_name, file_name)
    else:
        module = importlib.import_module(module_path_or_name)

    if module:
        name = module_path_or_name.split('.')[-1].split('/')[-1]

    return module, name


def _load_flags(*config_paths):
    """Aggregates gflags from `config_path` into global flags

    :param config_paths:
    :return:
    """
    for config_path in config_paths:
        print 'loading flags from', config_path
        _import_module(config_path)


def parse_flags():
    f = tf_flags.FLAGS
    args = sys.argv[1:]

    old_flags = f.__dict__['__flags'].copy()
    # Parse the known flags from that list, or from the command
    # line otherwise.
    flags_passthrough = f._parse_flags(args=args)  # pylint: disable=protected-access
    sys.argv[1:] = flags_passthrough
    f.__dict__['__flags'].update(old_flags)

    return f.__flags # pylint: disable=protected-access


def _restore_flags(flags):
    tf_flags.FLAGS.__dict__['__flags'] = flags
    tf_flags.FLAGS.__dict__['__parsed'] = True


def print_flags():
    flags = tf_flags.FLAGS.__flags

    print 'Flags:'
    keys = sorted(flags.keys())
    print '=' * 60
    for k in keys:
        print '\t{}: {}'.format(k, flags[k])
    print '=' * 60
    print


def set_flags(**flag_dict):
    for k, v in flag_dict.iteritems():
       sys.argv.append('--{}={}'.format(k, v))


def assert_all_flags_parsed():
    not_parsed = [a for a in sys.argv[1:] if a.startswith('--')]
    if not_parsed:
        raise RuntimeError('Failed to parse following flags: {}'.format(not_parsed))


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()


def set_flags_if_notebook(**flags_to_set):
    if is_notebook() and flags_to_set:
        print 'Setting the following flags:'
        keys = sorted(flags_to_set.keys())
        for k in keys[:-1]:
            print ' --{}={}\\'.format(k, flags_to_set[k])

        k = keys[-1]
        print ' --{}={}'.format(k, flags_to_set[k])

        set_flags(**flags_to_set)


def is_notebook():
    notebook = False
    try:
        interpreter = get_ipython().__class__.__name__
        if interpreter == 'ZMQInteractiveShell':
            notebook = True
        elif interpreter != 'TerminalInteractiveShell':
            raise ValueError('Unknown interpreter name: {}'.format(interpreter))

    except NameError:
        # get_ipython is undefined => no notebook
        pass
    return notebook


def optimizer_from_string(opt_string, build=True):
    import tensorflow as tf

    res = re.search(r'([a-z|A-Z]+)\(?(.*)\)?$', opt_string).groups()
    opt_name = res[0]

    opt_args = ''
    if len(res) > 1:
        opt_args = res[1]

    if opt_args.endswith(')'):
        opt_args = opt_args[:-1]

    opt_args = eval('dict({})'.format(opt_args))
    opt = getattr(tf.train, '{}Optimizer'.format(opt_name))

    if not build:
        opt = opt, opt_args
    else:
        opt = opt(**opt_args)
    return opt


def format_integer(number, group_size=3):
    assert group_size > 0

    number = str(number)
    parts = []

    while number:
        number, part = number[:-group_size], number[-group_size:]
        parts.append(part)

    number = ' '.join(reversed(parts))
    return number


def print_num_params():
    num_params = sum([np.prod(v.shape.as_list(), dtype=int) for v in tf.trainable_variables()])
    num_params = format_integer(num_params)
    print 'Number of trainable parameters: {}'.format(num_params)


def print_variables_by_scope():
    vars = [(v.name, v.shape.as_list()) for v in tf.trainable_variables()]
    vars = sorted(vars, key=lambda x: x[0])

    last_scope = None
    scope_n_params = 0
    for i, (name, shape) in enumerate(vars):

        current_scope = name.split('/', 1)[0]
        if current_scope != last_scope or i == len(vars) - 1:
            if last_scope is not None:
                scope_n_params = format_integer(scope_n_params)
                print '{} scope params = {}'.format(last_scope, scope_n_params)
                print

            print 'scope:', current_scope
            scope_n_params = 0

        last_scope = current_scope
        n_params = np.prod(shape, dtype=np.int32)
        scope_n_params += n_params
        print '\t', name, shape
    print


def get_session(tfdbg=False):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    if tfdbg:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    return sess


if __name__ == '__main__':

    tf_flags.DEFINE_integer('int_flag', -2, 'some int')
    tf_flags.DEFINE_string('string_flag', 'abc', 'some string')

    checkpoint_dir = '../checkpoints/setup'
    data_config = 'configs/static_mnist_data.py'
    model_config = 'configs/imp_weighted_nvil.py'


    # sys.argv.append('--int_flag=100')
    # sys.argv.append('--model_flag=-1')
    # print sys.argv

    experiment_folder, loaded_flags, checkpoint_dir = init_checkpoint(checkpoint_dir, data_config, model_config, resume=False)

    print experiment_folder
    print loaded_flags
    print checkpoint_dir
    print sys.argv

    print
    print 'tf.flags:'
    for k, v in tf_flags.FLAGS.__flags.iteritems():
        print k, v
    # batch_size = 64
    # data_dict = load(data_config, batch_size)
    # print data_dict.keys()
    #
    # model, train_step, global_step = load(model_config, img=data_dict.train_img, num=data_dict.train_num)
    #
    # print model
