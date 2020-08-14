"""
compiler.py - The compiler.py module contains the Compiler class
definition.
"""
import os
from tensorflow import keras


class Compiler:
    """
    Compiler - The Compiler class encapsulates logic related to
    compiling the model.
    """

    learning_rate = float(os.getenv("LEARNING_RATE", "1e-5"))
    if learning_rate <= 0 or learning_rate > 1:
        raise ValueError(
            "check LEARNING_RATE env var: learning rate " +
            "must be an integer value between (0-1]")

    @staticmethod
    def compile(bert_model: keras.Model):
        """

        :param bert_model:
        :type bert_model:
        :return:
        :rtype:
        """
        compiled_model: keras.Model = bert_model.compile(
            optimizer=keras.optimizers.Adam(
                Compiler.learning_rate
            ),
            loss=keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(
                    name="acc"
                )
            ]
        )
        return compiled_model
