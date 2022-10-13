import pytest
from kedro.runner import SequentialRunner
from kedro.io import DataCatalog

@pytest.fixture
def data():
    '''Only needed because DataPreLoader adds additional parameter to a test function self.
    If pytest does not find any fixture with the additional parameter then an error will be cast.
    '''
    pass


@pytest.fixture
def sequential_runner():
    return SequentialRunner()


@pytest.fixture
def catalog():
    return DataCatalog()
