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

    @classmethod
    def save_model_to_disk(cls, bert_model):
        """

        :param bert_model:
        :type bert_model:
        :param saved_model_path:
        :type saved_model_path:
        :return:
        :rtype:
        """
        keras.models.save_model(bert_model, cls.MODEL_DIR)
