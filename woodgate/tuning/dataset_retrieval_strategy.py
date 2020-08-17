"""
dataset_retrieval_strategy.py - The
dataset_retrieval_strategy.py module contains the
DatasetRetrievalStrategy class definition.
"""
import gdown
from ..build.file_system_configuration import \
    FileSystemConfiguration


class DatasetRetrievalStrategy:
    """
    DatasetRetrievalStrategy - The
    DatasetRetrievalStrategy class encapsulates logic
    related to collecting data required for fine tuning.
    """

    @staticmethod
    def retrieve_training_dataset(url: str) -> None:
        """This method will retrieve the dataset residing at the
        provided URL (:param url:). The implementation currently
        uses `gdown` and so it is implied the URL be pointing to a
        file located in either Google Drive or Dropbox.
        What makes this method specific to the training dataset is
        that the output is automatically stored on the host file
        system at `$TRAINING_PATH`. When the
        `woodgate.woodgate_process.WoodgateProcess.run()` method
        is executed there is an assumption made related to file
        structure and calling this method to retrieve the training
        dataset ensures the correct file will be used for
        training the learning model.

        :param url: URL string pointing to training dataset
        located on Google Drive or Dropbox.
        :type url: str
        :return: None
        :rtype: NoneType
        """
        gdown.download(
            url,
            output=FileSystemConfiguration.training_path
        )

        return None

    @staticmethod
    def retrieve_testing_dataset(url: str) -> None:
        """This method will retrieve the dataset residing at the
        provided URL (:param url:). The implementation currently
        uses `gdown` and so it is implied the URL be pointing to a
        file located in either Google Drive or Dropbox.
        What makes this method specific to the testing dataset is
        that the output is automatically stored on the host file
        system at `$TESTING_PATH`. When the
        `woodgate.woodgate_process.WoodgateProcess.run()` method
        is executed there is an assumption made related to file
        structure and calling this method to retrieve the testing
        dataset ensures the correct file will be used for testing
        the learning model.

        :param url: URL string pointing to testing dataset
        located on Google Drive or Dropbox.
        :type url: str
        :return: None
        :rtype: NoneType
        """
        gdown.download(
            url,
            output=FileSystemConfiguration.testing_path
        )

        return None

    @staticmethod
    def retrieve_evaluation_dataset(url: str) -> None:
        """This method will retrieve the dataset residing at the
        provided URL (:param url:). The implementation currently
        uses `gdown` and so it is implied the URL be pointing to a
        file located in either Google Drive or Dropbox.
        What makes this method specific to the evaluation dataset
        is that the output is automatically stored on the host
        file system at `$EVALUATION_PATH`. When the
        `woodgate.woodgate_process.WoodgateProcess.run()` method
        is executed there is an assumption made related to file
        structure and calling this method to retrieve the
        evaluation dataset ensures the correct file will be used
        for evaluating the learning model.

        :param url: URL string pointing to evaluation dataset
        located on Google Drive or Dropbox.
        :type url: str
        :return: None
        :rtype: NoneType
        """
        gdown.download(
            url,
            output=FileSystemConfiguration.evaluation_path
        )

        return None

    @staticmethod
    def retrieve_regression_dataset(url: str) -> None:
        """This method will retrieve the dataset residing at the
        provided URL (:param url:). The implementation currently
        uses `gdown` and so it is implied the URL be pointing to a
        file located in either Google Drive or Dropbox.
        What makes this method specific to the regression dataset
        is that the output is automatically stored on the host
        file system at `$REGRESSION_PATH`. When the
        `woodgate.woodgate_process.WoodgateProcess.run()` method
        is executed there is an assumption made related to file
        structure and calling this method to retrieve the
        regression dataset ensures the correct file will be used
        for evaluating the learning model.

        :param url: URL string pointing to regression dataset
        located on Google Drive or Dropbox.
        :type url: str
        :return: None
        :rtype: NoneType
        """
        gdown.download(
            url,
            output=FileSystemConfiguration.regression_path
        )

        return None
