import yaml
from tests.utils.data_loader import LoaderInterface

'''NOTE:
I stopped implementing this. It has not been tested.
'''
class LoaderNumpy(LoaderInterface):
    @staticmethod
    def load(path):
        def read_file(path):
            with open(path, "r") as stream:
                try:
                     return yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)
                    raise exc

        def create_key_from_path(path):
            return str(path).split('/')[-1].split('.')[0]

        _dict = read_file(path)
        key = create_key_from_path(path)

        return {key: _dict}
