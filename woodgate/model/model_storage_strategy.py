"""
model_storage_strategy.py - The model_storage_strategy.py module contains the ModelStorage class which encapsulates logic related to
persisting the model after fine tuning.
"""
from ..build.build_configuration import BuildConfiguration
from tensorflow import keras


class ModelStorageStrategy:
    """
    ModelStorage - The ModelStorage class encapsulates logic related to
    persisting the model after fine tuning.
    """

    def __init__(self, build_configuration: BuildConfiguration):
        """

        :param build_configuration:
        """
        self.model_build_dir = build_configuration.model_build_dir

    def save_model_to_disk(self, bert_model: keras.Model):
        """

        :param bert_model:
        :return:
        """
        keras.models.save_model(bert_model, self.model_build_dir)
