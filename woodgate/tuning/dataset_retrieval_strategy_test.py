"""
dataset_retrieval_strategy_test.py - The
dataset_retrieval_strategy_test.py module contains
unit tests for the dataset_retrieval_strategy.py
module.
"""
import os
import unittest
import shutil
from .dataset_retrieval_strategy import DatasetRetrievalStrategy
from ..tuning.external_datasets import ExternalDatasets
from ..woodgate_settings import Model, Build, FileSystem


class TestDatasetRetrieval(unittest.TestCase):
    """
    TestDatasetRetrieval class contains the unit tests related
    to the DatasetRetrieval class.
    """

    def setUp(self) -> None:
        """

        :return:
        """
        model = Model("test")
        build = Build()
        file_system = FileSystem(model, build)
        file_system.configure()
        self.file_system = file_system

    def tearDown(self) -> None:
        """

        :return:
        """
        # shutil.rmtree(self.file_system.woodgate_base_dir)

    def test_dataset_retrieval(self) -> None:
        """

        :return:
        :rtype:
        """
        self.assertFalse(
            os.path.isfile(
                self.file_system.get_training_path()
            )
        )

        DatasetRetrievalStrategy.retrieve_tuning_dataset(
            url=ExternalDatasets.training_dataset_url,
            output=self.file_system.get_training_path()
        )

        self.assertTrue(
            os.path.isfile(
                self.file_system.get_training_path()
            )
        )


if __name__ == '__main__':
    unittest.main()
