import os
from typing import List


def find_folder_by_dir_keyword(root_dir: str,
                               keyword: str) -> List[str]:
    """
    Parameters:
    -----------
    root_dir: The folder where the search begins.
    keyword: The text to search for within the desired dir. 

    Returns:
        A list of all folders found that contain the keyword within its dir.
    """

    def _get_all_sub_dirs(dir):
        return [x[0] for x in os.walk(dir)]      

    def _filter_by_keyword(dirs, keyword):
        return [dir for dir in dirs if keyword in dir]

    dirs = _get_all_sub_dirs(root_dir)
    dirs = _filter_by_keyword(dirs, keyword)

    return dirs
