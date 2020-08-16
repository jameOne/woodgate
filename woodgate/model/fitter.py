"""
fitter.py - The fitter.py module contains the Fitter
class definition.
"""
import os
from tensorflow import keras
from ..build.file_system_configuration import \
    FileSystemConfiguration
from ..fine_tuning.text_processor import TextProcessor


class Fitter:
    """
    Fitter - The Fitter class encapsulates logic related to
    fitting the model to the training data.
    """

    #: The `create_tensorboard_logs` attribute represents a
    #: signal variable that is used to decide whether
    #: tensorboard logs should be generated along with build
    #: logs. This attribute is set via the
    #: `CREATE_TENSORBOARD_LOGS` environment
    #: variable. If the `CREATE_TENSORBOARD_LOGS` environment
    #: variable is not set, then the `create_tensorboard_logs`
    #: attribute is set to `1` by default signaling the
    #: program to generate tensorboard logs. All values
    #: except `CREATE_TENSORBOARD_LOGS=0` signal the program
    #: to generate tensorboard logs.
    create_tensorboard_logs = int(
        os.getenv("CREATE_TENSORBOARD_LOGS", "1")
    )

    #: The `validation_split` attribute represents a decimal
    #: number between 0 and 1. This value indicates
    #: the proportional split of your training set by the
    #: value of the variable. For example, a value of
    #: `VALIDATION_SPLIT=0.2` would signal the program to
    #: reserve 20% of the training set for validation testing
    #: completed after each training epoch. If the
    #: `VALIDATION_SPLIT` environment variable is not set,
    #: then the `validation_split` attribute will default to
    #: `0.1`.
    validation_split: float = float(
        os.getenv("VALIDATION_SPLIT", "0.1")
    )
    if validation_split < 0 or validation_split > 1:
        raise ValueError(
            "check VALIDATION_SPLIT env var: validation "
            + "split must be a floating point value between "
            + "[0-1]"
        )

    #: The `batch_size` attribute represents an integer number
    #: between 8 and 512 inclusive. This value indicates the
    #: number of training examples utilized in one iteration.
    #: The batch size is a characteristic of gradient descent
    #: training algorithms. If the `BATCH_SIZE` environment
    #: variable is not set, then the `batch_size` attribute
    #: will default to `16`.
    batch_size: int = int(os.getenv("BATCH_SIZE", "16"))
    if batch_size < 8 or batch_size > 512:
        raise ValueError(
            "check BATCH_SIZE env var: batch size "
            + "batch size must be an integer value between "
            + "[8-512]"
        )

    #: The `epochs` attribute represents an integer between
    #: 1-1000 inclusive. This value indicates the number of
    #: times the training algorithm will iterate over the
    #: training dataset before completing. If the `EPOCHS`
    #: environment variable is unset, then the `epochs`
    #: attribute will default to `5`.
    epochs: int = int(os.getenv("EPOCHS", "5"))
    if epochs < 1 or epochs > 1000:
        raise ValueError(
            "check EPOCHS env var: epochs "
            + "must be an integer value between [0-1000]"
        )

    @classmethod
    def fit(
            cls,
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
        if cls.create_tensorboard_logs:
            callbacks.append(
                keras.callbacks.TensorBoard(
                    log_dir=FileSystemConfiguration.log_dir
                )
            )

        build_history = bert_model.fit(
            x=data.train_x,
            y=data.train_y,
            validation_split=cls.validation_split,
            batch_size=cls.batch_size,
            epochs=cls.epochs,
            callbacks=callbacks
        )

        return build_history
