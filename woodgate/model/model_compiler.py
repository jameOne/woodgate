"""
model_compiler.py - The model_compiler.py module contains the ModelCompiler class definition.
"""
import os
from tensorflow import keras


class ModelCompiler:
    """
    ModelCompiler - The ModelCompiler class encapsulates logic related to compiling the model.
    """

    LEARNING_RATE = os.getenv("LEARNING_RATE", "1e-5")
    try:
        LEARNING_RATE = float(LEARNING_RATE)
    except ValueError:
        LEARNING_RATE = 1e-5

    @staticmethod
    def compile(bert_model):
        """

        :param bert_model:
        :type bert_model:
        :return:
        :rtype:
        """
        compiled_model = bert_model.compile(
            optimizer=keras.optimizers.Adam(ModelCompiler.LEARNING_RATE),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
        )
        return compiled_model
