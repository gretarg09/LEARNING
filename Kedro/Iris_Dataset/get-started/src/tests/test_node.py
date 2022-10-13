import pytest
from tests.utils.data_loader import (DataPreLoader,
                                     LoaderPandas)
from tests.utils.dir import (find_files_by_dir_keyword,
                             get_root_of_test_data)

from get_started.nodes import split_data

root = get_root_of_test_data()

@DataPreLoader(paths=find_files_by_dir_keyword(root_dir=root,
                                               keyword='split_data',
                                               file_types=['.csv']),
               loader=LoaderPandas)
def test_split_data(data):
    parameters = {'random_state': 8,
                  'target_column': 'species',
                  'train_fraction': 0.9}

    result = split_data(data=data['input_data'],
                        parameters=parameters)
    
    assert len(result[0]) == 9
    assert len(result[1]) == 1
    assert len(result[2]) == 9
    assert len(result[3]) == 1
