"""Project settings. There is no need to edit this file unless you want to change values
from the Kedro defaults. For further information, including these default values, see
https://kedro.readthedocs.io/en/stable/kedro_project_setup/settings.html."""

from kedro.config import ConfigLoader, TemplatedConfigLoader
from get_started.utils.config.config_handler import IrisConfigLoader

ConfigLoader.__subclasses__ = lambda: [TemplatedConfigLoader]

CONFIG_LOADER_CLASS = IrisConfigLoader
