"""
fitter.py - The fitter.py module contains the Fitter
class definition.
"""
from typing import Type
from tensorflow import keras
from ..woodgate_settings import \
    WoodgateSettings
from ..tuning.text_processor import TextProcessor


class Fitter:
    """
    Fitter - The Fitter class encapsulates logic related to
    fitting the model to the training data.
    """

    def __init__(
            self,
            woodgate_settings: Type[WoodgateSettings]
    ):
        self.create_tensorboard_logs: int = woodgate_settings\
            .create_tensorboard_logs
        self.validation_split: float = woodgate_settings\
            .validation_split
        self.batch_size: int = woodgate_settings.batch_size
        self.epochs: int = woodgate_settings.epochs

    def fit(
            self,
            bert_model: keras.Model,
            data: TextProcessor
    ) -> keras.callbacks.History:
        """This method wraps the `fit` method of the `keras.Model`
        object argument which returns an object representing the
        history of the build.
        
        :param bert_model: The application specific (trained) \
        BERT model.
        :type bert_model: keras.Model
        :param data: Processed textual data.
        :type data: TextProcessor
        :return: A `History` object. Its `History.history` \
        attribute is a record of training loss values and \
        metrics values at successive epochs, as well as \
        validation loss values and validation metrics values \
        (if applicable).
        :rtype: object
        """
        
        callbacks = list()
        if self.create_tensorboard_logs:
            callbacks.append(
                keras.callbacks.TensorBoard(
                    log_dir=WoodgateSettings.log_dir
                )
            )

        build_history = bert_model.fit(
            x=data.train_x,
            y=data.train_y,
            validation_split=self.validation_split,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks
        )

        return build_history
