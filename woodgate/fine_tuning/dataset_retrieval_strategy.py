"""
dataset_retrieval_strategy.py - The
dataset_retrieval_strategy.py module contains the
FineTuningDataCollectionStrategy class definition.
"""
import gdown
from ..build.file_system_configuration import \
    FileSystemConfiguration


class DatasetRetrievalStrategy:
    """
    FineTuningDataCollectionStrategy - The
    FineTuningDataCollectionStrategy class encapsulates logic
    related to collecting data required for fine tuning.
    """

    @staticmethod
    def retrieve_training_dataset(url: str):
        """

        :param url:
        :type url:
        :return:
        :rtype:
        """
        gdown.download(
            url,
            output=FileSystemConfiguration.training_path
        )

    @staticmethod
    def retrieve_testing_dataset(url: str):
        """

        :param url:
        :type url:
        :return:
        :rtype:
        """
        gdown.download(
            url,
            output=FileSystemConfiguration.testing_path
        )

    @staticmethod
    def retrieve_evaluation_dataset(url: str):
        """

        :param url:
        :type url:
        :return:
        :rtype:
        """
        gdown.download(
            url,
            output=FileSystemConfiguration.evaluation_path
        )

    @staticmethod
    def retrieve_regression_dataset(url: str):
        """

        :param url:
        :type url:
        :return:
        :rtype:
        """
        gdown.download(
            url,
            output=FileSystemConfiguration.regression_path
        )
