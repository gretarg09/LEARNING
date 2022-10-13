import os
from os.path import isfile, join
from typing import List
import functools
import operator

from tests.utils.dir.find_folder_by_dir_keyword import find_folder_by_dir_keyword


def find_files_by_dir_keyword(root_dir: str,
                              keyword: str,
                              file_types: List[str]=None) -> List[str]:
    """
    Parameters:
    -----------
    root_dir: The folder where the search begins.
    keyword: The text to search for within the desired dir. 
    file_types: Specifies what file types should be in scope.

    Returns:
    --------
        A list of all files in scope within directories that contain keyword within its dir.
    """

    def _get_all_files_within_dir(dir):
        return [join(dir, file) for file in os.listdir(dir) if isfile(join(dir, file))]

    def _flatten_list(files):
        return functools.reduce(operator.iconcat, files, [])

    def _filter_by_file_types(files, file_types):  
        return filter(lambda item: any([item for file_type in file_types if file_type in item]),
                      files) 

    dirs = find_folder_by_dir_keyword(root_dir, keyword)
    files = map(_get_all_files_within_dir, dirs)
    files = _flatten_list(files)

    if file_types:
        files = _filter_by_file_types(files, file_types)

    return list(files)
