import pandas as pd 
from tests.utils.data_loader import LoaderInterface


class LoaderPandas(LoaderInterface):

    @staticmethod
    def load(path):
        def read_file(path):
            return pd.read_csv(path)

        def create_key_from_path(path):
            return str(path).split('/')[-1].split('.')[0]

        df = read_file(path)
        key = create_key_from_path(path)

        return {key: df}
