"""
definition_test.py - The definition_test.py module
contains all unit tests related to the definition.py module.
"""
import os
import unittest
import glob
import shutil
from tensorflow import keras
import pandas as pd
from ..transfer.bert_model_parameters import BertModelParameters
from ..transfer.bert_retrieval_strategy import \
    BertRetrievalStrategy
from ..woodgate_settings import WoodgateSettings
from .definition import Definition
from ..tuning.text_processor import TextProcessor
from ..tuning.external_datasets import ExternalDatasets


class TestDefinition(unittest.TestCase):
    """
    TestDefinition class contains the unit tests related to the
    Definition class.
    """
    def setUp(self) -> None:
        """

        :return:
        :rtype:
        """
        test_dir = os.path.join(
            os.path.dirname(__file__),
            "test_files"
        )
        os.makedirs(test_dir, exist_ok=True)

        bert_model_parameters = BertModelParameters(
            bert_h_param=128,
            bert_l_param=2
        )

        bert_retrieval_strategy = BertRetrievalStrategy(
            bert_model_parameters=bert_model_parameters
        )

        bert_retrieval_strategy.download_bert()

    def tearDown(self) -> None:
        """

        :return:
        :rtype:
        """
        test_dir = os.path.join(
            os.path.dirname(__file__),
            "test_files"
        )
        shutil.rmtree(test_dir)

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

    def test_create_model(self) -> None:
        """

        :return:
        :rtype:
        """
        os.makedirs(
            WoodgateSettings.testing_dir,
            exist_ok=True
        )

        os.makedirs(
            WoodgateSettings.training_dir,
            exist_ok=True
        )

        test = pd.DataFrame({
            "text": [
                "test intent"
            ],
            "intent": [
                "TestIntent"
            ]
        })
        with open(
                WoodgateSettings.get_testing_path(), "w+"
        ) as file:
            file.write(test.to_csv(index_label=False))
        ExternalDatasets.set_testing_data()

        train = pd.DataFrame({
            "text": [
                "train intent"
            ],
            "intent": [
                "TrainIntent"
            ]
        })
        with open(
                WoodgateSettings.get_training_path(), "w+"
        ) as file:
            file.write(train.to_csv(index_label=False))
        ExternalDatasets.set_training_data()

        data = TextProcessor(
            train,
            test,
            Definition.get_tokenizer(),
            [
                "TrainIntent",
                "TestIntent"
            ]
        )
        test_model = Definition.create_model(
            data.max_sequence_length,
            2
        )

        self.assertTrue(isinstance(test_model, keras.Model))


if __name__ == '__main__':
    unittest.main()
