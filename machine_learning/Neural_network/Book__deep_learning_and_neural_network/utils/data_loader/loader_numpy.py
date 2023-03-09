import numpy as np
from utils.data_loader import LoaderInterface


class LoaderNumpy(LoaderInterface):

    @staticmethod
    def load(path):
        def read_file(path):
            data = []

            for row in open(path):
                row = list(np.fromstring(row, sep=","))
                data.append(row)

            return data

        def create_array(list_of_array):
            return np.array(list_of_array)

        def create_key_from_path(path):
            return str(path).split('/')[-1].split('.')[0]

        list_of_array = read_file(path)
        array = create_array(list_of_array)
        key = create_key_from_path(path)

        return {key: array}
