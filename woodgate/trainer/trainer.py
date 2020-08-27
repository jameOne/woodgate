"""
trainer.py - The trainer.py module contains the
Fitter class definition.
"""
import os
import json
from bert.loader import (
    StockBertConfig,
    map_stock_config_to_params,
    load_stock_weights
)
import tensorflow as tf
from tensorflow import keras
from bert import BertModelLayer
from woodgate.transfer.bert_model_parameters import \
    BertModelParameters
from woodgate.woodgate_settings import (
    Architecture,
    FileSystem
)
from woodgate.tuning.external_datasets import ExternalDatasets
from woodgate.trainer.preprocessor import Preprocessor


class Trainer:
    """
    Trainer - The Trainer class encapsulates logic related to
    fitting the evaluator to the training data.
    """
    def __init__(
            self,
            validation_split: float,
            batch_size: int,
            epochs: int,
            file_system: FileSystem = None
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

        self.file_system = file_system

    @staticmethod
    def model_factory(
            name: str,
            external_datasets: ExternalDatasets,
            preprocessor: Preprocessor,
            architecture: Architecture,
            file_system: FileSystem,
    ) -> keras.Model:
        """The create_model method is a helper which accepts
        max input sequence length and the number of intents
        (classification bins/buckets). The logic returns a
        BERT evaluator that matches the specified architecture.

        :param name:
        :type name:
        :param external_datasets:
        :type external_datasets:
        :param preprocessor:
        :type preprocessor:
        :param architecture:
        :type architecture:
        :param file_system:
        :type file_system:
        :return:
        :rtype:
        """

        with tf.io.gfile.GFile(
                file_system.get_bert_config_path()) as reader:
            bc = StockBertConfig.from_json_string(
                reader.read()
            )
            bert_params = map_stock_config_to_params(bc)
            bert_params.adapter_size = None
            bert = BertModelLayer.from_params(
                bert_params,
                name=name
            )

        input_ids = keras.layers.Input(
            shape=(preprocessor.max_sequence_length,),
            dtype='int32',
            name="input_ids"
        )
        bert_output = bert(input_ids)

        clf_out = keras.layers.Lambda(
            lambda seq: seq[:, 0, :]
        )(bert_output)
        clf_out = keras.layers.Dropout(
            architecture.clf_out_dropout_rate
        )(clf_out)
        logits = keras.layers.Dense(
            units=BertModelParameters().bert_h_param,
            activation=architecture.clf_out_activation
        )(clf_out)
        logits = keras.layers.Dropout(
            architecture.logits_dropout_rate
        )(logits)
        logits = keras.layers.Dense(
            units=len(external_datasets.all_intents()),
            activation=architecture.logits_activation
        )(logits)

        model = keras.Model(
            inputs=input_ids,
            outputs=logits
        )
        model.build(
            input_shape=(None, preprocessor.max_sequence_length)
        )

        load_stock_weights(
            bert,
            file_system.get_bert_model_path()
        )

        return model

    def fit(
            self,
            bert_model: keras.Model,
            data: Preprocessor
    ) -> keras.callbacks.History:
        """This method wraps the `fit` method of the `keras.Model`
        object argument which returns an object representing the
        history of the build_history.

        :param bert_model: The application specific (trained) \
        BERT evaluator.
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
        if self.file_system is not None:
            callbacks.append(
                keras.callbacks.TensorBoard(
                    log_dir=self.file_system.log_dir
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

    @staticmethod
    def create_build_history_json(
            build_history: keras.callbacks.History,
            file_system: FileSystem,
    ) -> None:
        """The `create_loss_over_epochs_json` method creates a
        json document on the host file system in the
        `WoodgateSettings.build_summary_dir` directory.

        :param build_history:
        :type build_history:
        :param file_system:
        :type file_system:
        :return:
        :rtype:
        """

        build_history_path = os.path.join(
            file_system.build_dir,
            "buildHistory.json"
        )

        with open(build_history_path, "w+") as file:
            file.write(
                json.dumps(build_history.history)
            )

        return None
