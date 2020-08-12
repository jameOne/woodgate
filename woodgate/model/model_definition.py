"""
model_definition.py - This file contains the ModelDefinition class which encapsulates logic related to defining
the model layers.
"""
import os
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
import tensorflow as tf
from tensorflow import keras
from bert import BertModelLayer
from bert.tokenization.bert_tokenization import FullTokenizer
from woodgate.build.build_configuration import BuildConfiguration


class ModelDefinition:
    """
    ModelDefinition - Class - The ModelDefinition class encapsulates logic related to defining the model
    architecture.
    """

    def __init__(self, build_configuration: BuildConfiguration):
        """

        :param build_configuration:
        """

        #: The `bert_dir` attribute represents the path to a directory on the host file system containing the
        #: BERT model. This attribute is set via the `BERT_DIR` environment variable.
        #: For example, consider the following script:
        #: # Download BERT model
        #: wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip
        #:
        #: mkdir ~/models
        #: mkdir ~/models/bert
        #:
        #: # Unzip the file
        #: unzip uncased_L-12_H-768_A-12.zip -d ~/models/bert
        #:
        #: `~/models/bert` would be the bert_dir environment variable.
        #: If the `BERT_DIR` environment variable is not set, then the `bert_dir` attribute defaults to:
        #: `$WOODGATE_BASE_DIR/bert`. The program will attempt to create `BERT_DIR` if it does not already
        #: exist.
        self.bert_dir: str = os.getenv("BERT_DIR", os.path.join(build_configuration.woodgate_base_dir, "bert"))
        os.makedirs(self.bert_dir, exist_ok=True)

        self.bert_config: str = os.getenv("BERT_CONFIG", os.path.join(self.bert_dir, "bert_config.json"))

        self.bert_model: str = os.getenv("BERT_MODEL", os.path.join(self.bert_dir, "bert_model.ckpt"))

        self.tokenizer: FullTokenizer = FullTokenizer(
            vocab_file=os.path.join(self.bert_dir, "vocab.txt")
        )

    def create_model(self, max_sequence_length: int, number_of_intents: int):
        """
        ModelDefinition.create_model - Method - The create_model method is a helper which accepts
        max input sequence length and the number of intents (or bins/buckets). The logic returns a
        BERT model that matches the specified architecture.

        :param max_sequence_length: maximum length of input sequence
        :type max_sequence_length: int
        :param number_of_intents: number of classifiable bins/buckets
        :type number_of_intents: int
        :return: model definition
        :rtype: keras.Model
        """

        with tf.io.gfile.GFile(self.bert_config) as reader:
            bc = StockBertConfig.from_json_string(reader.read())
            bert_params = map_stock_config_to_params(bc)
            bert_params.adapter_size = None
            bert = BertModelLayer.from_params(bert_params, name="bert")

        input_ids = keras.layers.Input(shape=(max_sequence_length,), dtype='int32', name="input_ids")
        bert_output = bert(input_ids)

        cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
        cls_out = keras.layers.Dropout(0.5)(cls_out)
        logits = keras.layers.Dense(units=768, activation="tanh")(cls_out)
        logits = keras.layers.Dropout(0.5)(logits)
        logits = keras.layers.Dense(units=number_of_intents, activation="softmax")(logits)

        model: keras.Model = keras.Model(inputs=input_ids, outputs=logits)
        model.build(input_shape=(None, max_sequence_length))

        load_stock_weights(bert, self.bert_model)

        return model
