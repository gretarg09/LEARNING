import numpy as np
import pytest

from utils.data_loader import LoaderNumpy
from utils.dir import get_root_of_repository


@pytest.mark.loader_numpy
@pytest.mark.parametrize(
    'input_path, key, expected', [('tests/test_data/utils/data_loader/loader_numpy/input_1.csv',
                                   'input_1',
                                   [[0.033, 0.122], [1.33 , 8.88], [11.33 , 20.88]]),
                                  ('tests/test_data/utils/data_loader/loader_numpy/input_2.csv', 
                                   'input_2',
                                   [[0.033], [1.33], [11.33]])])
def test_loader_numpy(input_path, key, expected):
    test_file = get_root_of_repository() / input_path
    result = LoaderNumpy().load(test_file)
    expected = np.array(expected)

    assert (result[key] == expected).all()
