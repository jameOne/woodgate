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
from ..woodgate_settings import WoodgateSettings
from ..build.file_system_configuration import \
    FileSystemConfiguration


class TestDatasetRetrieval(unittest.TestCase):
    """
    TestDatasetRetrieval class contains the unit tests related
    to the DatasetRetrieval class.
    """

    def setUp(self) -> None:
        """

        :return:
        :rtype:
        """
        FileSystemConfiguration(
            woodgate_settings=WoodgateSettings
        )

    @classmethod
    def tearDownClass(cls) -> None:
        """

        :return:
        :rtype:
        """
        woodgate_base_dir = os.path.join(
            os.path.expanduser("~"),
            "woodgate"
        )
        shutil.rmtree(woodgate_base_dir)

    def test_dataset_retrieval(self) -> None:
        """

        :return:
        :rtype:
        """
        self.assertFalse(
            os.path.isfile(
                WoodgateSettings.get_training_path()
            )
        )

        DatasetRetrievalStrategy.retrieve_tuning_dataset(
            url=ExternalDatasets.training_dataset_url,
            output=WoodgateSettings.get_training_path()
        )

        self.assertTrue(
            os.path.isfile(
                WoodgateSettings.get_training_path()
            )
        )


if __name__ == '__main__':
    unittest.main()
