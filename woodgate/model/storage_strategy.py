"""
storage_strategy.py - The storage_strategy.py module
contains the StorageStrategy class which encapsulates logic
related to persisting the model after fine tuning.
"""
import tensorflow as tf
from tensorflow import keras
from ..woodgate_settings import WoodgateSettings


class StorageStrategy:
    """
    StorageStrategy - The StorageStrategy class encapsulates
    logig related to persisting the model after fine tuning.
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
        tf.saved_model.save(
            bert_model,
            WoodgateSettings.model_build_dir
        )

        return None

    @staticmethod
    def load_model() -> keras.Model:
        """The `load_model` method is a convenience method
        which wraps the `keras.models.load_model(...)` method,
        called with the `WoodgateSettings.model_build_dir`
        attribute as it's argument.

        :return: A `keras.Model` object loaded from file system.
        :rtype: keras.Model
        """
        loaded_model = keras.models.load_model(
            WoodgateSettings.model_build_dir
        )

        return loaded_model
