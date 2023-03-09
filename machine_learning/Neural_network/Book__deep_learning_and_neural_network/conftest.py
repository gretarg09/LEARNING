import pytest


@pytest.fixture
def data():
    '''Only needed because DataPreLoader adds additional parameter to a test function self.
    If pytest does not find any fixture with the additional parameter then an error will be cast.
    '''
    pass
