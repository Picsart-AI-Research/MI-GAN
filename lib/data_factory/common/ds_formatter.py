def singleton(class_):
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance


@singleton
class get_formatter(object):
    def __init__(self):
        self.formatter = {}

    def register(self, formatf):
        self.formatter[formatf.__name__] = formatf

    def __call__(self, cfg):
        if cfg is None:
            return None
        t = cfg.type
        return self.formatter[t](**cfg.args)


def register():
    def wrapper(class_):
        get_formatter().register(class_)
        return class_
    return wrapper
