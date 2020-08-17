"""
text_processor_test.py - Module - The text_processor_test.py
module contains all unit tests related to the text_processor.py
module.
"""
import shutil
import unittest
import subprocess
import pandas as pd
import numpy as np
from .text_processor import TextProcessor
from ..model.definition import Definition
from ..build.file_system_configuration import \
    FileSystemConfiguration


class TestTextProcessor(unittest.TestCase):
    """
    TestTextProcessor - This class contains unit tests related to
    the TextProcessor class.
    """

    def setUp(self) -> None:
        """
        wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-2_H-128_A-2.zip

        mkdir ./test
        mkdir ./test/bert

        # Unzip the file
        unzip uncased_L-2_H-128_A-2.zip -d ./test/bert/tiny

        rm uncased_L-2_H-128_A-2.zip

        :return:
        """

        process = subprocess.Popen(
            [
                'wget',
                'https://storage.googleapis.com/bert_models/'
                + '2020_02_20/uncased_L-2_H-128_A-2.zip'
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        _ = process.communicate()

        process = subprocess.Popen(
            [
                'unzip',
                'uncased_L-2_H-128_A-2.zip',
                '-d',
                FileSystemConfiguration.bert_dir
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        _ = process.communicate()

        process = subprocess.Popen(
            [
                'rm',
                'uncased_L-2_H-128_A-2.zip'
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        _ = process.communicate()

    def tearDown(self) -> None:
        """

        :return:
        """
        shutil.rmtree(FileSystemConfiguration.woodgate_base_dir)

    def test_text_processor(self) -> None:
        """

        :return:
        """

        FileSystemConfiguration()

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
