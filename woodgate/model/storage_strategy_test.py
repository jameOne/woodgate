"""
storage_strategy_test.py - The storage_strategy_test.py module
contains unit tests related to the woodgate.model.storage_strategy
module.
"""
import unittest
import os
import glob
import shutil
import pandas as pd
from tensorflow import keras
from ..tuning.external_datasets import ExternalDatasets
from ..tuning.text_processor import TextProcessor
from .definition import Definition
from .compiler import Compiler
from .fitter import Fitter
from .storage_strategy import StorageStrategy
from ..woodgate_settings import WoodgateSettings
from ..build.file_system_configuration import \
    FileSystemConfiguration
from ..transfer.bert_model_parameters import BertModelParameters
from ..transfer.bert_retrieval_strategy import \
    BertRetrievalStrategy


class TestStorageStrategy(unittest.TestCase):
    """
    TestStorageStrategy class contains unit tests related to the
    StorageStrategy class.
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

    def test_save_model(self) -> None:
        """

        :return:
        :rtype:
        """
        test = pd.DataFrame({
            "text": [
                "test intent john and test intent",
                "test intent jacob and test intent",
                "test intent jingle and  dfg test intent",
                "test intent sdf and test intent",
                "test intent and sdf test intent",
                "test intent and brp test intent",
                "test intent doug and test intent",
                "test intent and crow test intent",
                "test intent and all test intent",
                "test intent and test intent"
            ],
            "intent": [
                "TestIntent0",
                "TestIntent0",
                "TestIntent0",
                "TestIntent0",
                "TestIntent0",
                "TestIntent0",
                "TestIntent0",
                "TestIntent0",
                "TestIntent0",
                "TestIntent0"
            ]
        })
        with open(
                WoodgateSettings.get_testing_path(), "w+"
        ) as file:
            file.write(test.to_csv(index_label=False))
        ExternalDatasets.set_testing_data()

        with open(
                WoodgateSettings.get_training_path(), "w+"
        ) as file:
            file.write(test.to_csv(index_label=False))
        ExternalDatasets.set_training_data()

        data = TextProcessor(
            test,
            test,
            Definition.get_tokenizer(),
            [
                "TestIntent0",
            ]
        )
        test_model = Definition.create_model(
            data.max_sequence_length,
            1
        )

        Compiler.compile(test_model)

        fitter = Fitter(
            woodgate_settings=WoodgateSettings
        )

        _ = fitter.fit(
            test_model,
            data
        )

        StorageStrategy.save_model(test_model)

        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    WoodgateSettings.model_build_dir,
                    "saved_model.pb"
                )
            )
        )

        loaded_model = StorageStrategy.load_model()

        self.assertTrue(
            isinstance(loaded_model, keras.Model)
        )


if __name__ == '__main__':
    unittest.main()
