"""
preprocessor.py - This preprocessor.py module contains the
TextPreprocessor class definition.
"""
import os
import numpy as np
from bert.tokenization.bert_tokenization import FullTokenizer


class Preprocessor:
    """
    Preprocessor - The Preprocessor class encapsulates logic
    related to processing clean text (in CSV format) used to
    train BERT for intent detection.
    """

    data_column_title = os.getenv(
        "DATA_COLUMN_TITLE",
        "text"
    )
    label_column_title = os.getenv(
        "LABEL_COLUMN_TITLE",
        "intent"
    )

    def __init__(
            self,
            train,
            test,
            vocab_file: str,
            intents,
            max_sequence_length=128,
    ):
        self.tokenizer = self.tokenizer_factory(vocab_file)
        self.max_sequence_length = 0
        self.intents = intents
        (
            (self.train_x, self.train_y),
            (self.test_x, self.test_y)
        ) = map(self._prepare, [train, test])
        self.max_sequence_length = min(
            self.max_sequence_length,
            max_sequence_length
        )
        self.train_x, self.test_x = map(
            self._pad,
            [self.train_x, self.test_x]
        )

    def _prepare(self, df):
        x, y = [], []
        for _, row in df.iterrows():
            text, label = \
                row[self.data_column_title], \
                row[self.label_column_title]
            tokens = self.tokenizer.tokenize(text)
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            token_ids = self.tokenizer.convert_tokens_to_ids(
                tokens
            )
            self.max_sequence_length = max(
                self.max_sequence_length,
                len(token_ids)
            )
            x.append(token_ids)
            y.append(self.intents.index(label))
        return np.array(x), np.array(y)

    def _pad(self, ids):
        x = []
        for input_ids in ids:
            input_ids = input_ids[
                        :min(
                            len(input_ids),
                            self.max_sequence_length - 2)
                        ]
            input_ids += [0] * (
                    self.max_sequence_length - len(input_ids)
            )
            x.append(np.array(input_ids))
        return np.array(x)

    @staticmethod
    def tokenizer_factory(vocab_file: str) -> FullTokenizer:
        """This method will return a BERT tokenizer initialized
        using the vocabulary file at
        `WoodgateSettings.bert_vocab_path`.

        :return: A BERT tokenizer.
        :rtype: FullTokenizer
        """
        tokenizer: FullTokenizer = FullTokenizer(
            vocab_file=vocab_file
        )
        return tokenizer
