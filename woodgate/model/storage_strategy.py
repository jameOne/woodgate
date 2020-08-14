"""
storage_strategy.py - The storage_strategy.py module
contains the StorageStrategy class which encapsulates logic
related to persisting the model after fine tuning.
"""
from ..build.file_system_configuration import \
    FileSystemConfiguration
from tensorflow import keras


class StorageStrategy:
    """
    StorageStrategy - The StorageStrategy class encapsulates logic
    related to persisting the model after fine tuning.
    """

    @staticmethod
    def save_model(bert_model: keras.Model):
        """

        :param bert_model:
        :return:
        """
        keras.models.save_model(
            bert_model,
            FileSystemConfiguration.model_build_dir
        )
