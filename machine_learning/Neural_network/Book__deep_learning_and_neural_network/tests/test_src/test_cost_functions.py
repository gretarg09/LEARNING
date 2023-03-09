import pytest
import numpy as np

from src.cost_functions import Quadratic, CrossEntropy
from utils.data_loader import DataPreLoader, LoaderNumpy
from utils.dir import find_files_by_dir_keyword, get_root_of_test_data


class TestQuadratic:

    @pytest.mark.quadratic_cost
    @DataPreLoader(paths=find_files_by_dir_keyword(get_root_of_test_data(), 'cost_functions/quadratic/fn'),
                   loader=LoaderNumpy)
    def test_fn(self, data):
        result = Quadratic.fn(data['input_a'],
                              data['input_y'])

        assert result == 0.9380080000000001

    @pytest.mark.quadratic_cost
    @DataPreLoader(paths=find_files_by_dir_keyword(get_root_of_test_data(), 'cost_functions/quadratic/delta'),
                   loader=LoaderNumpy)
    def test_delta(self, data):
        result = Quadratic.delta(data['input_z'], 
                                 data['input_a'], 
                                 data['input_y'])

        assert np.isclose(result, data['expected']).all()


class TestCrossEntropy:

    @pytest.mark.cross_entropy
    @DataPreLoader(paths=find_files_by_dir_keyword(get_root_of_test_data(), 'cost_functions/cross_entropy/fn'),
                   loader=LoaderNumpy)
    def test_fn(self, data):
        result = CrossEntropy.fn(data['input_a'],
                                 data['input_y'])

        assert result == 6.715670270793827 

    @pytest.mark.cross_entropy
    @DataPreLoader(paths=find_files_by_dir_keyword(get_root_of_test_data(), 'cost_functions/cross_entropy/delta'),
                   loader=LoaderNumpy)
    def test_delta(self, data):
        result = CrossEntropy.delta(data['input_z'], 
                                    data['input_a'], 
                                    data['input_y'])

        assert np.isclose(result, data['expected']).all()
