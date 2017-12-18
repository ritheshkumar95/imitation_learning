import ops
import tensorflow as tf
import locale
import numpy as np

locale.setlocale(locale.LC_ALL, '')

_params = {}
_param_aliases = {}


def param(name, *args, **kwargs):
    """
    A wrapper for `tf.Variable` which enables parameter sharing in models.

    Creates and returns theano shared variables similarly to `tf.Variable`,
    except if you try to create a param with the same name as a
    previously-created one, `param(...)` will just return the old one instead of
    making a new one.

    This constructor also adds a `param` attribute to the shared variables it
    creates, so that you can easily search a graph for all params.
    """

    if name not in _params:
        kwargs['name'] = name
        param = tf.Variable(*args, **kwargs)
        param.param = True
        _params[name] = param

    return _params[name]


def delete_all_params():
    _params.clear()


def print_model_settings(locals_):
    print "Model settings:"
    all_vars = [(k, v) for (k, v) in locals_.items() if (k.isupper() and k != 'T')]
    all_vars = sorted(all_vars, key=lambda x: x[0])
    for var_name, var_value in all_vars:
        print "\t{}: {}".format(var_name, var_value)


def print_model_settings_dict(settings):
    print "Settings dict:"
    all_vars = [(k, v) for (k, v) in settings.items()]
    all_vars = sorted(all_vars, key=lambda x: x[0])
    for var_name, var_value in all_vars:
        print "\t{}: {}".format(var_name, var_value)


def print_params_info():
    params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    params = sorted(params, key=lambda p: p.name.split('/')[-1].split(':')[0])
    shapes = [p.shape.as_list() for p in params]

    total_param_count = 0
    log = "\nParams for cost:"
    for param, shape in zip(params, shapes):
        log += ("\n\t%-20s %s" % (shape, param.name.split('/')[-1].split(':')[0]))
        total_param_count += reduce(lambda a, b: a*b, shape)

    log += "\nTotal parameter count for this cost:\n\t{0}".format(
        locale.format("%d", total_param_count, grouping=True)
    )
    print log


def save_params(sess, path):
    weights = {}
    for param, variable in _params.iteritems():
        weights[param] = sess.run(variable.value())

    np.save(path, weights)


def load_params(sess, path):
    weights = np.load(path).tolist()
    for param, variable in _params.iteritems():
        sess.run(variable.assign(weights[param]))
    print "Weights Loaded!"
