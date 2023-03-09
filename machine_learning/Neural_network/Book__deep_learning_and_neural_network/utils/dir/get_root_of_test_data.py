from utils.dir.get_root_of_repository import get_root_of_repository


def get_root_of_test_data():
    return get_root_of_repository() / 'tests/test_data' 
