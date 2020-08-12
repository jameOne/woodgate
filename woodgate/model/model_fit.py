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

    def __init__(self, build_configuration: BuildConfiguration):
        #: The `create_tensorboard_logs` attribute represents a signal variable that is used
        #: to decide whether tensorboard logs should be generated along with build logs.
        #: This attribute is set via the `CREATE_TENSORBOARD_LOGs` environment
        #: variable. If the `CREATE_TENSORBOARD_LOGS` environment variable is not set, then
        #: the `create_tensorboard_logs` attribute is set to `1` by default signaling the program to generate
        #: tensorboard logs. All values except `CREATE_TENSORBOARD_LOGS=0` signal the program to generate
        #: tensorboard logs.
        self.create_tensorboard_logs = int(os.getenv("CREATE_TENSORBOARD_LOGS", "1"))

        #: The `validation_split` attribute represents a decimal number between 0 and 1. This value indicates
        #: the proportional split of your training set by the value of the variable. For example, a value of
        #: `VALIDATION_SPLIT=0.2` would signal the program to reserve 20% of the training set for validation testing
        #: completed after each training epoch. If the `VALIDATION_SPLIT` environment variable is not set, then the
        #: `validation_split` attribute will default to `0.1`.
        self.validation_split: float = float(os.getenv("VALIDATION_SPLIT", "0.1"))
        if self.validation_split < 0 or self.validation_split > 1:
            raise ValueError(
                "check VALIDATION_SPLIT env var: " +
                "validation split must be a floating point value between [0-1]")

        #: The `batch_size` attribute represents an integer number between 8 and 512 inclusive. This value indicates the
        #: number of training examples utilized in one iteration. The batch size is a characteristic of gradient descent
        #: training algorithms. If the `BATCH_SIZE` environment variable is not set, then the `batch_size` attribute
        #: will default to `16`.
        self.batch_size: int = int(os.getenv("BATCH_SIZE", "16"))
        if self.batch_size < 8 or self.batch_size > 512:
            raise ValueError(
                "check BATCH_SIZE env var: " +
                "batch size must be an integer value between [8-512]")

        #: The `epochs` attribute represents an integer between 1-1000 inclusive. This value indicates the number of
        #: times the training algorithm will iterate over the training dataset before completing. If the `EPOCHS`
        #: environment variable is unset, then the `epochs` attribute will default to `5`.
        self.epochs: int = int(os.getenv("EPOCHS", "5"))
        if self.epochs < 1 or self.epochs > 1000:
            raise ValueError(
                "check EPOCHS env var: " +
                "epochs must be an integer value between [0-1000]")

        self.log_dir = build_configuration.log_dir

    def fit(self, bert_model, data):
        """
        
        :param bert_model: 
        :type bert_model: 
        :param data: 
        :type data: 
        :return: 
        :rtype: 
        """
        
        callbacks = list()
        if self.create_tensorboard_logs:
            callbacks.append(keras.callbacks.TensorBoard(log_dir=self.log_dir))

        build_history = bert_model.fit(
            x=data.train_x,
            y=data.train_y,
            validation_split=self.validation_split,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks
        )

        return build_history
