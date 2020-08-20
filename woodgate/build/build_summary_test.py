"""
build_summary_test.py - The build_summary_test.py module
contains all unit tests related to the
woodgate.build.build_summary module.
"""
import os
import unittest
import glob
import shutil
import pandas as pd
from .build_summary import BuildSummary
from .file_system_configuration import FileSystemConfiguration
from ..woodgate_settings import WoodgateSettings
from ..transfer.bert_retrieval_strategy import \
    BertRetrievalStrategy
from ..transfer.bert_model_parameters import BertModelParameters
from ..tuning.external_datasets import ExternalDatasets
from ..tuning.text_processor import TextProcessor
from ..model.definition import Definition
from ..model.fitter import Fitter
from ..model.compiler import Compiler


class TestBuildSummary(unittest.TestCase):
    """
    TestBuildSummary class contains the unit tests related to
    the BuildSummary class.
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

    def test_create_loss_over_epochs_plot(self) -> None:
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

        test_build_history = fitter.fit(
            test_model,
            data
        )

        BuildSummary.create_loss_over_epochs_plot(
            build_history=test_build_history
        )

        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    WoodgateSettings.build_summary_dir,
                    "loss_over_epochs.png"
                )
            )
        )

    def test_create_acc_over_epochs_plot(self) -> None:
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

        test_build_history = fitter.fit(
            test_model,
            data
        )

        BuildSummary.create_accuracy_over_epochs_plot(
            build_history=test_build_history
        )

        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    WoodgateSettings.build_summary_dir,
                    "accuracy_over_epochs.png"
                )
            )
        )

    def test_create_acc_over_epochs_json(self) -> None:
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

        test_build_history = fitter.fit(
            test_model,
            data
        )

        BuildSummary.create_accuracy_over_epochs_json(
            build_history=test_build_history
        )

        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    WoodgateSettings.build_summary_dir,
                    "accuracyOverEpochs.json"
                )
            )
        )

    def test_create_loss_over_epochs_json(self) -> None:
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

        test_build_history = fitter.fit(
            test_model,
            data
        )

        BuildSummary.create_loss_over_epochs_json(
            build_history=test_build_history
        )

        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    WoodgateSettings.build_summary_dir,
                    "lossOverEpochs.json"
                )
            )
        )


if __name__ == '__main__':
    unittest.main()
