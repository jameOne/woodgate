"""
storage.py - The storage.py module
contains the StorageStrategy class which encapsulates logic
related to persisting the evaluator after fine tuning.
"""
from tensorflow import keras
from woodgate.woodgate_settings import FileSystem


class Storage:
    """
    StorageStrategy - The StorageStrategy class encapsulates
    logig related to persisting the evaluator after fine tuning.
    """

    @staticmethod
    def save_model(
            bert_model: keras.Model,
            file_system: FileSystem
    ) -> None:
        """

        :param bert_model:
        :type bert_model:
        :param file_system:
        :type file_system:
        :return:
        :rtype:
        """
        keras.models.save_model(
            bert_model,
            file_system.build_dir
        )

        return None

    @staticmethod
    def load_model(file_system: FileSystem) -> keras.Model:
        """The `load_model` method is a convenience method
        which wraps the `keras.models.load_model(...)` method,
        called with the `WoodgateSettings.model_build_dir`
        attribute as it's argument.

        :return: A `keras.Model` object loaded from file system.
        :rtype: keras.Model
        """
        loaded_model = keras.models.load_model(
            file_system.build_dir
        )

        return loaded_model
