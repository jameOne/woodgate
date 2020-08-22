"""
model_training.py - The model_training.py module contains the
Fitter class definition.
"""
from tensorflow import keras
from woodgate.preprocessor.preprocessor import Preprocessor


class Trainer:
    """
    Trainer - The Trainer class encapsulates logic related to
    fitting the model to the training data.
    """
    def __init__(
            self,
            validation_split: float,
            batch_size: int,
            epochs: int,
            use_tensorboard_callback: bool,
            log_dir: str
    ):
        """

        :param validation_split:
        :type validation_split:
        :param batch_size:
        :type batch_size:
        :param epochs:
        :type epochs:
        """
        #: The `validation_split` attribute represents a decimal
        #: number between 0 and 1. This attribute is set via the
        #: `VALIDATION_SPLIT` environment variable.
        #: Validation split indicates the proportional split of
        #: your training set by the value of the variable.
        #: For example, a value of `VALIDATION_SPLIT=0.2`
        #: would signal the program to reserve 20% of the
        #: training set for validation testing
        #: completed after each training epoch. If the
        #: `VALIDATION_SPLIT` environment variable is not set,
        #: then the `validation_split` attribute will default to
        #: `0.1`.
        self.validation_split: float = validation_split

        #: The `batch_size` attribute represents an integer number
        #: between 8 and 512 inclusive. This value indicates the
        #: number of training examples utilized in one iteration.
        #: The batch size is a characteristic of gradient descent
        #: training algorithms. If the `BATCH_SIZE` environment
        #: variable is not set, then the `batch_size` attribute
        #: will default to `16`.
        self.batch_size: int = batch_size

        #: The `epochs` attribute represents an integer between
        #: 1-1000 inclusive.
        #: This attribute is set via the `EPOCHS` environment
        #: variable. This value indicates the number of
        #: times the training algorithm will iterate over the
        #: training dataset before completing. If the `EPOCHS`
        #: environment variable is unset, then the `epochs`
        #: attribute will default to `5`.
        self.epochs: int = epochs

        #: The `use_tensorboard_callback` attribute represents a
        #: signal variable that is used to decide whether
        #: tensorboard logs should be generated along with build
        #: logs. This attribute is set via the
        #: `CREATE_TENSORBOARD_LOGS` environment
        #: variable. If the `CREATE_TENSORBOARD_LOGS` environment
        #: variable is not set, then the
        #: `use_tensorboard_callback` attribute is set to `1`
        #: by default signaling the program to generate
        #: tensorboard logs. All values except
        #: `CREATE_TENSORBOARD_LOGS=0` signal the program
        #: to generate tensorboard logs.
        self.use_tensorboard_callback: bool = \
            use_tensorboard_callback

        #: The `log_dir` attribute represents the path to a
        #: directory on the host file system where the
        #: log data will be stored. The log data directory should
        #: be a child directory of the build output directory
        #: in which the program will store the data retrieved for
        #: logging the build process. This attribute is set via
        #: the `LOG_DIR` environment variable. If the `LOG_DIR`
        #: environment variable is not set, then the `log_dir`
        #: attribute will default to `$OUTPUT_DIR/log`. The
        #: program will attempt to create `LOG_DIR` if it does
        #: not already exist.
        self.log_dir = log_dir

    def fit(
            self,
            bert_model: keras.Model,
            data: Preprocessor
    ) -> keras.callbacks.History:
        """This method wraps the `fit` method of the `keras.Model`
        object argument which returns an object representing the
        history of the build.

        :param bert_model: The application specific (trained) \
        BERT model.
        :type bert_model: keras.Model
        :param data: Processed textual data.
        :type data: Preprocessor
        :return: A `History` object. Its `History.history` \
        attribute is a record of training loss values and \
        metrics values at successive epochs, as well as \
        validation loss values and validation metrics values \
        (if applicable).
        :rtype: object
        """

        callbacks = list()
        if self.use_tensorboard_callback:
            callbacks.append(
                keras.callbacks.TensorBoard(
                    log_dir=self.log_dir
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
