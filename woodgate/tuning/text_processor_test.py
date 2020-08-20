"""
text_processor_test.py - Module - The text_processor_test.py
module contains all unit tests related to the text_processor.py
module.
"""
import os
import glob
import shutil
import unittest
import pandas as pd
import numpy as np
from .text_processor import TextProcessor
from ..model.definition import Definition
from ..woodgate_settings import WoodgateSettings
from ..build.file_system_configuration import \
    FileSystemConfiguration
from ..transfer.bert_model_parameters import BertModelParameters
from ..transfer.bert_retrieval_strategy import \
    BertRetrievalStrategy


class TestTextProcessor(unittest.TestCase):
    """
    TestTextProcessor - This class contains unit tests related to
    the TextProcessor class.
    """

    def setUp(self) -> None:
        """

        :return:
        :rtype:
        """
        FileSystemConfiguration(
            woodgate_settings=WoodgateSettings
        )

        bert_model_parameters = BertModelParameters(
            bert_h_param=128,
            bert_l_param=2
        )

        bert_retrieval_strategy = BertRetrievalStrategy(
            bert_model_parameters=bert_model_parameters
        )

        bert_retrieval_strategy.download_bert()

    @classmethod
    def tearDownClass(cls) -> None:
        """

        :return:
        :rtype:
        """
        woodgate_base_dir = os.path.join(
            os.path.expanduser("~"),
            "woodgate"
        )
        shutil.rmtree(woodgate_base_dir)

    def test_setup(self) -> None:
        """

        :return:
        :rtype:
        """
        self.assertTrue(
            glob.glob(
                f"{WoodgateSettings.get_bert_model_path()}*"
            )
        )

    def test_text_processor(self) -> None:
        """

        :return:
        """

        FileSystemConfiguration(
            woodgate_settings=WoodgateSettings
        )

        test = pd.DataFrame({
            "text": [
                "test intent"
            ],
            "intent": [
                "TestIntent"
            ]
        })

        train = pd.DataFrame({
            "text": [
                "train intent"
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

        self.assertEqual(text_processor.max_sequence_length, 4)
        self.assertListEqual(text_processor.intents, intents)
        self.assertTrue(
            np.array_equal(
                text_processor.train_x,
                np.array([[101, 3345]])
            )
        )
        self.assertTrue(
            np.array_equal(
                text_processor.train_y,
                np.array([1])
            )
        )


if __name__ == '__main__':
    unittest.main()
