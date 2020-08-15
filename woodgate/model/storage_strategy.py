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
    def save_model(bert_model: keras.Model) -> None:
        """This method wraps the `save_model` method of the
        argument. The method will save the model in the
        `FileSystemConfiguration.model_build_dir` directory.

        :param bert_model: Any Keras model.
        :type bert_model: keras.Model
        :return: None
        :rtype: NoneType
        """
        keras.models.save_model(
            bert_model,
            FileSystemConfiguration.model_build_dir
        )

        return None
