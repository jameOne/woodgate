"""
compiler.py - The compiler.py module contains the Compiler class
definition.
"""
from typing import List
from tensorflow import keras


class Compiler:
    """
    Compiler - The Compiler class encapsulates logic related to
    compiling the evaluator.
    """

    @staticmethod
    def optimizer_factory(
            name: str,
            learning_rate: float,
            *args
    ) -> keras.optimizers.Optimizer:
        """

        :param name:
        :type name:
        :param learning_rate:
        :type learning_rate:
        :param args:
        :type args:
        :return:
        :rtype:
        """
        # ensure the name is lower case before
        # selecting the return statement
        name = name.lower()

        if name == "adam":
            return keras.optimizers.Adam(
                learning_rate,
                *args
            )
        elif name == "adamax":
            return keras.optimizers.Adamax(
                learning_rate,
                *args
            )
        elif name == "adadelta":
            return keras.optimizers.Adadelta(
                learning_rate,
                *args
            )
        elif name == "adagrad":
            return keras.optimizers.Adagrad(
                learning_rate,
                *args
            )
        elif name == "ftrl":
            return keras.optimizers.Ftrl(
                learning_rate,
                *args
            )
        elif name == "sgd":
            return keras.optimizers.SGD(
                learning_rate,
                *args
            )
        elif name == "rmsprop":
            return keras.optimizers.RMSprop(
                learning_rate,
                *args
            )
        else:
            raise ValueError(
                "optimizer must be either: "
                + '"adam", "adamax", "adadelta", "adagrad", '
                + '"ftrl", "sgd", or "rmsprop"'
            )

    @staticmethod
    def loss_factory(
            name: str,
            *args
    ) -> keras.losses.Loss:
        """

        :param name:
        :type name:
        :param args:
        :type args:
        :return:
        :rtype:
        """
        # ensure the name is lower case before
        # selecting the return statement
        name = name.lower()

        if name == "binary_crossentropy":
            from_logits = bool(args[0])
            label_smoothing = float(args[1])
            if label_smoothing > 1 or label_smoothing < 0:
                raise ValueError(
                    "label_smoothing (second loss arg) "
                    + "must be a value [0, 1]"
                )
            return keras.losses.BinaryCrossentropy(
                from_logits=from_logits,
                label_smoothing=label_smoothing
            )
        elif name == "categorical_crossentropy":
            from_logits = bool(args[0])
            label_smoothing = float(args[1])
            if label_smoothing > 1 or label_smoothing < 0:
                raise ValueError(
                    "label_smoothing (second loss arg) "
                    + "must be a value [0, 1]"
                )
            return keras.losses.CategoricalCrossentropy(
                from_logits=from_logits,
                label_smoothing=label_smoothing
            )
        elif name == "categorical_hinge":
            return keras.losses.CategoricalHinge()
        elif name == "cosine_similarity":
            axis = int(args[0])
            return keras.losses.CosineSimilarity(
                axis=axis
            )
        elif name == "hinge":
            return keras.losses.Hinge()
        elif name == "huber":
            delta = float(args[0])
            return keras.losses.Huber(
                delta=delta
            )
        elif name == "kl_divergence":
            return keras.losses.KLDivergence()
        elif name == "log_cosh":
            return keras.losses.LogCosh()
        elif name == "mean_absolute_error":
            return keras.losses.MeanAbsoluteError()
        elif name == "mean_absolute_percentage_error":
            return keras.losses.MeanAbsolutePercentageError()
        elif name == "mean_squared_error":
            return keras.losses.MeanSquaredError()
        elif name == "mean_squared_logarithmic_error":
            return keras.losses.MeanSquaredLogarithmicError()
        elif name == "poisson":
            return keras.losses.Poisson()
        elif name == "sparse_categorical_crossentropy":
            from_logits = bool(args[0])
            return keras.losses.SparseCategoricalCrossentropy(
                from_logits=from_logits
            )
        else:
            raise ValueError(
                "loss must be either: "
                + '"binary_crossentropy", '
                + '"categorical_crossentropy", '
                + '"categorical_hinge", '
                + '"cosine_similarity", '
                + '"hinge", '
                + '"huber", '
                + '"kl_divergence", '
                + '"log_cosh", '
                + '"mean_absolute_error", '
                + '"mean_absolute_percentage_error", '
                + '"mean_squared_error", '
                + '"mean_squared_logarithmic_error", '
                + '"poisson", or, '
                + '"sparse_categorical_crossentropy"'
            )

    @staticmethod
    def metrics_factory(*args) -> List[keras.metrics.Metric]:
        """

        :param args:
        :type args:
        :return:
        :rtype:
        """

        # ensure the name is lower case before
        # selecting the metrics_list.append(...)
        # statement
        args = [arg.lower() for arg in args]

        metrics_list = list()

        if "binary_crossentropy" in args:
            metrics_list.append(
                keras.metrics.BinaryCrossentropy()
            )
        if "binary_accuracy" in args:
            metrics_list.append(
                keras.metrics.BinaryAccuracy()
            )
        if "categorical_crossentropy" in args:
            metrics_list.append(
                keras.metrics.CategoricalCrossentropy()
            )
        if "categorical_hinge" in args:
            metrics_list.append(
                keras.metrics.CategoricalHinge()
            )
        if "cosine_similarity" in args:
            metrics_list.append(
                keras.metrics.CosineSimilarity()
            )
        if "hinge" in args:
            metrics_list.append(
                keras.metrics.Hinge()
            )
        if "kl_divergence" in args:
            metrics_list.append(
                keras.metrics.KLDivergence()
            )
        if "mean_absolute_error" in args:
            metrics_list.append(
                keras.metrics.MeanAbsoluteError()
            )
        if "mean_absolute_percentage_error" in args:
            metrics_list.append(
                keras.metrics.MeanAbsolutePercentageError()
            )
        if "mean_squared_error" in args:
            metrics_list.append(
                keras.metrics.MeanSquaredError()
            )
        if "mean_squared_logarithmic_error" in args:
            metrics_list.append(
                keras.metrics.MeanSquaredLogarithmicError()
            )
        if "poisson" in args:
            metrics_list.append(
                keras.metrics.Poisson()
            )
        if "sparse_categorical_accuracy" in args:
            metrics_list.append(
                keras.metrics.SparseCategoricalAccuracy()
            )
        if "sparse_categorical_crossentropy" in args:
            metrics_list.append(
                keras.metrics.SparseCategoricalCrossentropy()
            )
        if "accuracy" in args:
            metrics_list.append(
                keras.metrics.Accuracy()
            )
        if "auc" in args:
            metrics_list.append(
                keras.metrics.AUC()
            )

        return metrics_list

    @classmethod
    def compile(
            cls,
            model: keras.Model,
            optimizer: keras.optimizers.Optimizer,
            loss: keras.losses.Loss,
            metrics: List[keras.metrics.Metric]
    ) -> None:
        """This method will call the `compile` method on the
        `keras.Model` setting the optimizer, the loss function
        and various metrics.

        :param model:
        :type model:
        :param optimizer:
        :type optimizer:
        :param loss:
        :type loss:
        :param metrics:
        :type metrics:
        :return:
        :rtype:
        """

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

        return None
