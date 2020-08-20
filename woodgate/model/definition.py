"""
definition.py - This file contains the Definition class which
encapsulates logic related to defining the model layers.
"""
from bert.loader import (
    StockBertConfig,
    map_stock_config_to_params,
    load_stock_weights
)
import tensorflow as tf
from tensorflow import keras
from bert import BertModelLayer
from bert.tokenization.bert_tokenization import FullTokenizer
from ..woodgate_settings import WoodgateSettings
from ..transfer.bert_model_parameters import BertModelParameters


class Definition:
    """
    Definition - Class - The Definition class encapsulates logic
    related to defining the model architecture.
    """

    @staticmethod
    def get_tokenizer() -> FullTokenizer:
        """This method will return a BERT tokenizer initialized
        using the vocabulary file at
        `WoodgateSettings.bert_vocab_path`.

        :return: A BERT tokenizer.
        :rtype: FullTokenizer
        """
        tokenizer: FullTokenizer = FullTokenizer(
            vocab_file=WoodgateSettings.get_bert_vocab_path()
        )
        return tokenizer

    @staticmethod
    def create_model(
            max_sequence_length: int,
            number_of_intents: int
    ):
        """
        The create_model method is a helper which accepts
        max input sequence length and the number of intents
        (classification bins/buckets). The logic returns a
        BERT model that matches the specified architecture.

        :param max_sequence_length: max length of input sequence
        :type max_sequence_length: int
        :param number_of_intents: number of bins/buckets
        :type number_of_intents: int
        :return: model definition
        :rtype: keras.Model
        """

        with tf.io.gfile.GFile(
                WoodgateSettings.get_bert_config_path()
        ) as reader:
            bc = StockBertConfig.from_json_string(reader.read())
            bert_params = map_stock_config_to_params(bc)
            bert_params.adapter_size = None
            bert = BertModelLayer.from_params(
                bert_params,
                name="bert"
            )

        input_ids = keras.layers.Input(
            shape=(max_sequence_length,),
            dtype='int32',
            name="input_ids"
        )
        bert_output = bert(input_ids)

        cls_out = keras.layers.Lambda(
            lambda seq: seq[:, 0, :])(bert_output)
        cls_out = keras.layers.Dropout(0.5)(cls_out)
        logits = keras.layers.Dense(
            units=BertModelParameters().bert_h_param,
            activation="tanh"
        )(cls_out)
        logits = keras.layers.Dropout(0.5)(logits)
        logits = keras.layers.Dense(
            units=number_of_intents, activation="softmax")(logits)

        model: keras.Model = keras.Model(
            inputs=input_ids, outputs=logits)
        model.build(input_shape=(None, max_sequence_length))

        load_stock_weights(
            bert,
            WoodgateSettings.get_bert_model_path()
        )

        return model
