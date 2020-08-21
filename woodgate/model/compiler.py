"""
compiler.py - The compiler.py module contains the Compiler class
definition.
"""
from tensorflow import keras
from ..factories.optimizer_factory import OptimizerFactory
from ..factories.loss_factory import LossFactory
from ..factories.metrics_factory import MetricsFactory
from ..woodgate_settings import WoodgateSettings


class Compiler:
    """
    Compiler - The Compiler class encapsulates logic related to
    compiling the model.
    """

    @staticmethod
    def compile(bert_model: keras.Model) -> None:
        # TODO - Open a number of options for loss, optimizer,
        #  and metrics.
        """This method will call the `compile` method on the
        `keras.Model` setting the optimizer, the loss function
        and the metrics.

        :param bert_model: The BERT transfer model gathered \
        from Google.
        :type bert_model: keras.Model
        :return: None
        :rtype: NoneType
        """
        optimizer = OptimizerFactory.get_optimizer(
            WoodgateSettings.optimizer_name,
            WoodgateSettings.optimizer_learning_rate,
            *WoodgateSettings.optimizer_args
        )

        loss = LossFactory.get_loss(
            WoodgateSettings.loss_name,
            *WoodgateSettings.loss_args
        )

        metrics = MetricsFactory.get_metrics(
            WoodgateSettings.optimizer_metrics
        )

        bert_model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

        return None
