"""
model_fit.py - The model_fit.py module contains the ModelFit class definition.
"""
import os
from tensorflow import keras
from ..build.build_configuration import BuildConfiguration


class ModelFit:
    """
    ModelFit - The ModelFit class encapsulates logic related to fitting the model to
    the training data.
    """

    TENSORBOARD = os.getenv("TENSORBOARD", "1")
    try:
        TENSORBOARD = int(TENSORBOARD)
    except ValueError:
        TENSORBOARD = 1

    @staticmethod
    def fit(bert_model, data):
        """
        
        :param bert_model: 
        :type bert_model: 
        :param data: 
        :type data: 
        :return: 
        :rtype: 
        """
        
        callbacks = list()
        if ModelFit.TENSORBOARD:
            callbacks.append(keras.callbacks.TensorBoard(log_dir=BuildConfiguration.LOG_DIR))

        build_history = bert_model.fit(
            x=data.train_x,
            y=data.train_y,
            validation_split=BuildConfiguration.VALIDATION_SPLIT,
            batch_size=BuildConfiguration.BATCH_SIZE,
            epochs=BuildConfiguration.EPOCHS,
            callbacks=callbacks
        )

        return build_history
