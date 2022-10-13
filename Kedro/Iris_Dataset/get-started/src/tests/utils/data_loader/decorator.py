from functools import wraps
from collections import ChainMap


class DataPreLoader:
    """ Test data loader
    """

    def __init__(self, paths, loader):
        self.paths = paths
        self.loader = loader
    
    def __call__(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            print('running decorator')
            data = map(self.loader.load, self.paths)
            data = self.change_to_dict(list(data))

            kwargs['data'] = data

            result = fn(*args, **kwargs)
            return result

        return wrapper 

    @staticmethod
    def change_to_dict(data):
        return dict(ChainMap(*data))
