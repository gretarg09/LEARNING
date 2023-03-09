import pytest
import numpy as np

from utils.data_loader import DataPreLoader, LoaderNumpy
from utils.dir import find_files_by_dir_keyword, get_root_of_test_data


@pytest.mark.data_pre_loader
@DataPreLoader(paths=find_files_by_dir_keyword(get_root_of_test_data(), 'data_loader/decorator'),
               loader=LoaderNumpy)
def test_data_pre_loader(data):
    expected = np.array([(0.033, 0.122), (1.33 , 8.88), (11.33 , 20.88)])

    assert (data['input'] == expected).all()
