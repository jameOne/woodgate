"""
model_summary.py - The model_summary.py module contains the ModelSummary class definition.
"""
from tensorflow import keras


class ModelSummary:
    """
    ModelSummary - The ModelSummary class encapsulates logic related to summarizing the model.
    """

    @staticmethod
    def summarize(bert_model: keras.Model):
        """

        :param bert_model:
        :type bert_model:
        :return:
        :rtype:
        """
        return bert_model.summary()
