"""
model_storage.py - The model_storage.py module contains the ModelStorage class which encapsulates logic related to
persisting the model after fine tuning.
"""
import os
from ..build.build_configuration import BuildConfiguration
from tensorflow import keras


class ModelStorage:
    """
    ModelStorage - Class - The ModelStorage class encapsulates logic related to
    persisting the model after fine tuning.
    """

    MODEL_DIR = os.path.join(BuildConfiguration.OUTPUT_DIR, BuildConfiguration.MODEL_NAME)

    @staticmethod
    def save_model_to_disk(bert_model, model_dir):
        """

        :param bert_model:
        :param model_dir:
        :return:
        """
        keras.models.save_model(bert_model, model_dir)
