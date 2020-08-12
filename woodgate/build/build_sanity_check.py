"""
build_sanity_check.py - The build_sanity_check.py module contains the SanityCheck class definition.
"""
from tensorflow import keras

from ..model.model_storage import ModelStorageStrategy
from ..model.model_evaluation import ModelEvaluation
from ..fine_tuning.fine_tuning_text_processor import FineTuningTextProcessor


class BuildSanityCheck:
    """
    BuildSanityCheck - The BuildSanityCheck class encapsulates logic related
    to ensuring the completed build has resulted in a usable machine
    learning model.
    """

    def __init__(self, model_storage_strategy: ModelStorageStrategy):
        """

        :param model_storage_strategy:
        """
        self.model_build_dir = model_storage_strategy.model_build_dir

    def check_sanity(self, data: FineTuningTextProcessor):
        """

        :param data:
        :type data:
        :return:
        :rtype:
        """
        loaded_bert_model = keras.models.load_model(self.model_build_dir)
        ModelEvaluation.perform_regression_testing(loaded_bert_model, data)
