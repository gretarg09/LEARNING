import pytest
#from kedro.runner import SequentialRunner
from kedro.io import MemoryDataSet
from tests.utils.data_loader import (DataPreLoader,
                                     LoaderPandas)
from tests.utils.dir import (find_files_by_dir_keyword,
                             get_root_of_test_data)

from get_started.pipeline import create_pipeline


root = get_root_of_test_data()

class TestGetStartedPipeline:
    @DataPreLoader(paths=find_files_by_dir_keyword(root_dir=root,
                                                   keyword='pipeline',
                                                   file_types=['.csv']),
                   loader=LoaderPandas)
    def test_pipeline(self,
                      sequential_runner,
                      catalog,
                      data):

        catalog.add("example_iris_data", MemoryDataSet(data['input']))
        catalog.add_feed_dict({'parameters': {'random_state': 8,
                                              'target_column': 'species',
                                              'train_fraction': 0.9}})

        pipeline = (create_pipeline().from_inputs('example_iris_data')
                                     .to_outputs('accuracy'))

        output = sequential_runner.run(pipeline, catalog) 

        assert output['accuracy'] > 0.86
        assert output['accuracy'] < 0.87
