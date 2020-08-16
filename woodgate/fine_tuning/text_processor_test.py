"""
text_processor_test.py - Module - The text_processor_test.py
module contains all unit tests related to the text_processor.py
module.
"""
import unittest
import pandas as pd
from .text_processor import TextProcessor
from ..model.definition import Definition
from bert.tokenization.bert_tokenization import FullTokenizer


class TestTextProcessor(unittest.TestCase):
    """
    TestTextProcessor - This class contains unit tests related to
    the TextProcessor class.
    """

    def test_text_processor(self) -> None:
        """

        :return:
        """

        test = pd.DataFrame({
            "text": [
                "test"
            ],
            "intent": [
                "TestIntent"
            ]
        })

        train = pd.DataFrame({
            "text": [
                "train"
            ],
            "intent": [
                "TrainIntent"
            ]
        })

        tokenizer = Definition.get_tokenizer()

        intents = [
            "TestIntent",
            "TrainIntent"
        ]

        text_processor = TextProcessor(
            train=train,
            test=test,
            tokenizer=tokenizer,
            intents=intents
        )

        self.assertListEqual(text_processor.intents, intents)


if __name__ == '__main__':
    unittest.main()
