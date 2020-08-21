"""
loss_factory.py - The loss_factory.py module contains the
LossFactory class definition.
"""
from tensorflow.keras import losses


class LossFactory:
    """
    LossFactory class encapsulates logic related to
    generating the `loss` to be used during the model
    compilation process.
    """

    @staticmethod
    def get_loss(
            name: str,
            *loss_args
    ) -> losses.Loss:
        """

        :param name:
        :type name:
        :param loss_args:
        :type loss_args:
        :return:
        :rtype:
        """
        # ensure the name is lower case before
        # selecting the return statement
        name = name.lower()

        if name == "binary_crossentropy":
            from_logits = bool(loss_args[0])
            label_smoothing = float(loss_args[1])
            if label_smoothing > 1 or label_smoothing < 0:
                raise ValueError(
                    "label_smoothing (second loss arg) "
                    + "must be a value [0, 1]"
                )
            return losses.BinaryCrossentropy(
                from_logits=from_logits,
                label_smoothing=label_smoothing
            )
        elif name == "categorical_crossentropy":
            from_logits = bool(loss_args[0])
            label_smoothing = float(loss_args[1])
            if label_smoothing > 1 or label_smoothing < 0:
                raise ValueError(
                    "label_smoothing (second loss arg) "
                    + "must be a value [0, 1]"
                )
            return losses.CategoricalCrossentropy(
                from_logits=from_logits,
                label_smoothing=label_smoothing
            )
        elif name == "categorical_hinge":
            return losses.CategoricalHinge()
        elif name == "cosine_similarity":
            axis = int(loss_args[0])
            return losses.CosineSimilarity(
                axis=axis
            )
        elif name == "hinge":
            return losses.Hinge()
        elif name == "huber":
            delta = float(loss_args[0])
            return losses.Huber(
                delta=delta
            )
        elif name == "kl_divergence":
            return losses.KLDivergence()
        elif name == "log_cosh":
            return losses.LogCosh()
        elif name == "mean_absolute_error":
            return losses.MeanAbsoluteError()
        elif name == "mean_absolute_percentage_error":
            return losses.MeanAbsolutePercentageError()
        elif name == "mean_squared_error":
            return losses.MeanSquaredError()
        elif name == "mean_squared_logarithmic_error":
            return losses.MeanSquaredLogarithmicError()
        elif name == "poisson":
            return losses.Poisson()
        elif name == "sparse_categorical_crossentropy":
            from_logits = bool(loss_args[0])
            return losses.SparseCategoricalCrossentropy(
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
