"""
optimizer_factory.py - The optimizer_factory.py module contains
the OptimizerFactory class definition.
"""
from tensorflow import keras


class OptimizerFactory:
    """
    OptimizerFactory class encapsulates logic related to
    generating the `optimizer` to be used during the model
    compilation process.
    """

    @staticmethod
    def get_optimizer(
            name: str,
            learning_rate: float,
            *optimizer_args
    ) -> keras.optimizers.Optimizer:
        """

        :param name:
        :type name:
        :param learning_rate:
        :type learning_rate:
        :param optimizer_args:
        :type optimizer_args:
        :return:
        :rtype:
        """
        # ensure the name is lower case before
        # selecting the return statement
        name = name.lower()

        if name == "adam":
            return keras.optimizers.Adam(
                learning_rate,
                *optimizer_args
            )
        elif name == "adamax":
            return keras.optimizers.Adamax(
                learning_rate,
                *optimizer_args
            )
        elif name == "adadelta":
            return keras.optimizers.Adadelta(
                learning_rate,
                *optimizer_args
            )
        elif name == "adagrad":
            return keras.optimizers.Adagrad(
                learning_rate,
                *optimizer_args
            )
        elif name == "ftrl":
            return keras.optimizers.Ftrl(
                learning_rate,
                *optimizer_args
            )
        elif name == "sgd":
            return keras.optimizers.SGD(
                learning_rate,
                *optimizer_args
            )
        elif name == "rmsprop":
            return keras.optimizers.RMSprop(
                learning_rate,
                *optimizer_args
            )
        else:
            raise ValueError(
                "optimizer must be either: "
                + '"adam", "adamax", "adadelta", "adagrad", '
                + '"ftrl", "sgd", or "rmsprop"'
            )
