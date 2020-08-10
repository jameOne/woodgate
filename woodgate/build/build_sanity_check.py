"""
build_sanity_check.py - The build_sanity_check.py module contains the SanityCheck class definition.
"""
from tensorflow import keras

from ..model.model_storage import ModelStorage
from ..model.model_evaluation import ModelEvaluation


class BuildSanityCheck:
    """
    BuildSanityCheck - The BuildSanityCheck class encapsulates logic related
    to ensuring the completed build has resulted in a usable machine
    learning model.
    """

    @staticmethod
    def check_sanity(data):
        """

        :param data:
        :type data:
        :return:
        :rtype:
        """
        loaded_bert_model = keras.models.load_model(ModelStorage.MODEL_DIR)
        ModelEvaluation.perform_regression_testing(loaded_bert_model, data)
