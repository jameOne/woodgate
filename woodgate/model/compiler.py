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

    #: The `learning_rate` attribute represents a floating point
    #: value which represents the step size taken by the
    #: optimization algorithm toward a minimum loss. This value
    #: should be in the interval (0-1], otherwise a ValueError
    #: will be thrown at run time.
    learning_rate = float(os.getenv("LEARNING_RATE", "1e-5"))
    if learning_rate <= 0 or learning_rate > 1:
        raise ValueError(
            "check LEARNING_RATE env var: learning rate " +
            "must be an integer value between (0-1]")

    @staticmethod
    def compile(bert_model: keras.Model) -> keras.Model:
        # TODO - Open a number of options for loss, optimizer, and
        #   metrics.
        """This method will call the `compile` method on the
        `keras.Model` setting the optimizer, the loss function and
        the metrics.

        :param bert_model: The BERT transfer model gathered from \
        Google.
        :type bert_model: keras.Model
        :return: A BERT model compiled from the transfer model \
        (having optimizer, loss function, and metrics set) ready \
        for training.
        :rtype: keras.Model
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
