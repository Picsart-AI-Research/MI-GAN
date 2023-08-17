from importlib import import_module


def get_experiment(stagename):
    major, minor = stagename.split('.')
    mod = import_module('.experiments.'+major, package='lib')
    met = getattr(mod, minor)
    return met
