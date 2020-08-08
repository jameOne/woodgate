"""
text_processor.py - This file contains the Preprocessor class which encapsulates logic related to
processing clean text for intent detection by BERT.
"""
import os
import tqdm
import numpy as np
from bert.tokenization.bert_tokenization import FullTokenizer


class TextProcessor:
    """
    TextProcessor - Class which encapsulates utilities related to processing clean text for intent
    detection by BERT.
    """
    DATA_COLUMN = os.getenv("DATA_COLUMN", "text")
    LABEL_COLUMN = os.getenv("LABEL_COLUMN", "intent")

    def __init__(
            self,
            train,
            test,
            tokenizer: FullTokenizer,
            intents,
            max_sequence_length=128
    ):
        self.tokenizer = tokenizer
        self.max_sequence_length = 0
        self.intents = intents
        ((self.train_x, self.train_y), (self.test_x, self.test_y)) = \
            map(self._prepare, [train, test])
        print("max seq_len", self.max_sequence_length)
        self.max_sequence_length = min(self.max_sequence_length, max_sequence_length)
        self.train_x, self.test_x = map(
            self._pad,
            [self.train_x, self.test_x]
        )

    def _prepare(self, df):
        x, y = [], []
        for _, row in tqdm.tqdm(df.iterrows()):
            text, label = \
                row[TextProcessor.DATA_COLUMN], \
                row[TextProcessor.LABEL_COLUMN]
            tokens = self.tokenizer.tokenize(text)
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            self.max_sequence_length = max(self.max_sequence_length, len(token_ids))
            x.append(token_ids)
            y.append(self.intents.index(label))
        return np.array(x), np.array(y)

    def _pad(self, ids):
        x = []
        for input_ids in ids:
            input_ids = input_ids[:min(len(input_ids), self.max_sequence_length - 2)]
            input_ids += [0] * (self.max_sequence_length - len(input_ids))
            x.append(np.array(input_ids))
        return np.array(x)
