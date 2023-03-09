from pathlib import Path  


def get_root_of_repository(path=None):
    if not path:
        path = __file__

    for path in Path(path).parents:
        git_dir = path / '.git'
        if git_dir.is_dir():
            return path

    raise ValueError('The path provided nor __file__ is within a git repository.') 
