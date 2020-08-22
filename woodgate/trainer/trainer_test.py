"""
model_training_test.py - The model_training_test.py module contains all
unit tests related to the woodgate.model.fitter module.
"""
import os
import glob
import unittest
import shutil
import pandas as pd
from ..build.file_system_configuration import \
    FileSystemConfiguration
from woodgate.woodgate_settings import WoodgateSettings
from woodgate.transfer.bert_model_parameters import BertModelParameters
from woodgate.transfer.bert_retrieval_strategy import \
    BertRetrievalStrategy
from woodgate.tuning.external_datasets import ExternalDatasets
from woodgate.preprocessor.preprocessor import Preprocessor
from woodgate.model.definition import Definition
from woodgate.compiler.compiler import Compiler
from .model_training import Fitter


class TestFitter(unittest.TestCase):

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

    def test_fit_w_tensorboard_callback(self) -> None:
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

        with open(
                WoodgateSettings.get_evaluation_path(), "w+"
        ) as file:
            file.write(test.to_csv(index_label=False))
        ExternalDatasets.set_evaluation_data()

        with open(
                WoodgateSettings.get_regression_path(), "w+"
        ) as file:
            file.write(test.to_csv(index_label=False))
        ExternalDatasets.set_regression_data()

        data = Preprocessor(
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

        build_history = fitter.fit(
            test_model,
            data
        )

        self.assertIsNotNone(build_history)

    def test_fit_wo_tensorboard_callback(self) -> None:
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

        with open(
                WoodgateSettings.get_evaluation_path(), "w+"
        ) as file:
            file.write(test.to_csv(index_label=False))
        ExternalDatasets.set_evaluation_data()

        with open(
                WoodgateSettings.get_regression_path(), "w+"
        ) as file:
            file.write(test.to_csv(index_label=False))
        ExternalDatasets.set_regression_data()

        data = Preprocessor(
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

        WoodgateSettings.create_tensorboard_logs = 0

        fitter = Fitter(
            woodgate_settings=WoodgateSettings
        )

        build_history = fitter.fit(
            test_model,
            data
        )

        self.assertIsNotNone(build_history)

        WoodgateSettings.create_tensorboard_logs = 1


if __name__ == '__main__':
    unittest.main()
