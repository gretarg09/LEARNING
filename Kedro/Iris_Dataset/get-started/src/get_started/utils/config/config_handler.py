import os
import yaml
from typing import Dict, Any, Optional
from kedro.config import TemplatedConfigLoader
from get_started.utils.config import config_loader


class IrisConfigLoader(TemplatedConfigLoader):
    def __init__(self,
                 conf_source: str,
                 env: str = None,
                 runtime_params: Dict[str, Any] = None,
                 *,
                 base_env: str = "base",
                 default_run_env: str = "local",
                 globals_pattern: Optional[str] = "*globals.yml",
                 globals_dict: Optional[Dict[str, Any]] = None):


        print('\n----------Inputs----------\n')
        print('conf_source ',conf_source)
        print('env: ',env)
        print('runtime params: ', runtime_params)

        config = load_test_config()

        yaml_globals = config['globals']
        yaml_parameters = config['parameters']
        yaml_catalog = config['catalog']

        folder_path = create_folder(path=conf_source, name=yaml_globals['customer_id'])

        for object, name in [(yaml_globals, 'globals.yaml'),
                             (yaml_parameters, 'parameters.yaml'),
                             (yaml_catalog, 'catalog.yaml')]:

           safe_yaml(object=object, path=folder_path, name=name)

        default_run_env = str(yaml_globals['customer_id'])


        super().__init__(conf_source,
                         env,
                         runtime_params,
                         base_env=base_env,
                         default_run_env=default_run_env,
                         globals_pattern=globals_pattern,
                         globals_dict=globals_dict,)

def create_folder(path, name):
    path = os.path.join(path, str(name))
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def safe_yaml(object, path, name):
    path = os.path.join(path, str(name))
    with open(path, 'w') as file:
        yaml.dump(object, file)

def load_test_config():
    path = '/home/gretar_atli/Git/LEARNING__kedro/Iris_Dataset/get-started/src/'
    filename = 'test_config.yaml'

    return config_loader.load(path +filename)
