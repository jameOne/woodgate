"""
sanity_check.py - The sanity_check.py module contains
the SanityCheck class definition.
"""
from tensorflow import keras

from ..model.evaluation import ModelEvaluation
from ..tuning.text_processor import TextProcessor
from .file_system_configuration import FileSystemConfiguration


class SanityCheck:
    """
    SanityCheck - The SanityCheck class encapsulates
    logic related to ensuring the completed build has resulted in
    a usable machine learning model.
    """

    @staticmethod
    def check_sanity(data: TextProcessor) -> None:
        """This method will attempt to load a model from the
        `$MODEL_BUILD_DIR` and re-run regression testing. Thus it
        is expected this method is called after fine tuning BERT
        and saving the resulting model to `$MODEL_BUILD_DIR`.
        This test simply ensures the model saved to the file
        system behaves identically to the model resulting from
        fine tuning the BERT model (the one that was regression
        tested while in memory).

        :param data: Accepts a `TextProcessor` object, where the \
        `TextProcessor` represents the processed textual data \
        used to train the machine learning model.
        :type data: TextProcessor
        :return: None
        :rtype: NoneType
        """
        ModelEvaluation.perform_regression_testing(
            keras.models.load_model(
                FileSystemConfiguration.model_build_dir
            ),
            data
        )

        return None
