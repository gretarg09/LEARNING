from utils.dir import get_root_of_repository, find_folder_by_dir_keyword


def test_find_folder_by_dir_keyword():
    
     test_data_root = get_root_of_repository() / 'tests/test_data'
     result = find_folder_by_dir_keyword(root_dir=test_data_root, 
                                         keyword='find_folder_by_dir_keyword')

     assert result[0].split('/')[-1] == 'find_folder_by_dir_keyword'

    
