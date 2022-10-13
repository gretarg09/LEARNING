import os
from tests.utils.dir.get_root_of_repository import get_root_of_repository


def get_root_of_test_data():
    return os.path.join(get_root_of_repository(),
                        'Iris_Dataset/get-started/src/tests/test_data/')
