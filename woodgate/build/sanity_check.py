"""
sanity_check.py - The sanity_check.py module contains
the SanityCheck class definition.
"""
from tensorflow import keras

from ..model.evaluation import ModelEvaluation
from ..fine_tuning.text_processor import TextProcessor
from .file_system_configuration import FileSystemConfiguration


class SanityCheck:
    """
    SanityCheck - The SanityCheck class encapsulates
    logic related to ensuring the completed build has resulted in
    a usable machine learning model.
    """

    @staticmethod
    def check_sanity(data: TextProcessor):
        """

        :param data:
        :type data:
        :return:
        :rtype:
        """
        loaded_bert_model = keras.models.load_model(
            FileSystemConfiguration.model_build_dir
        )
        ModelEvaluation.perform_regression_testing(
            loaded_bert_model,
            data
        )
