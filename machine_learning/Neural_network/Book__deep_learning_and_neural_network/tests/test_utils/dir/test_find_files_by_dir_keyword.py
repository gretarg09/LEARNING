import pytest
from utils.dir import find_files_by_dir_keyword, get_root_of_repository


@pytest.mark.parametrize('file_types, expected', [(None, ['a.csv', 'b.csv', 'c.py', 'd.sh']),
                                                  (['.csv'], ['a.csv', 'b.csv']),
                                                  (['.py'], ['c.py']),
                                                  (['.sh'], ['d.sh'])])
def test_find_files_by_dir_keyword(file_types, expected):
    test_data_root = get_root_of_repository() / 'tests/test_data'

    result = [file.split('/')[-1] for file in find_files_by_dir_keyword(root_dir=test_data_root,
                                                                        keyword='find_files_by_dir_keyword',
                                                                        file_types=file_types)]

    assert sorted(result) == sorted(expected)

