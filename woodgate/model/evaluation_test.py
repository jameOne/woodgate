"""
evaluation_test.py - The evaluation_test.py module
contains all unit tests related to the evaluation.py module.
"""
import os
import unittest
import glob
import shutil
import pandas as pd
from ..woodgate_settings import WoodgateSettings
from ..transfer.bert_model_parameters import BertModelParameters
from ..transfer.bert_retrieval_strategy import \
    BertRetrievalStrategy
from ..build.file_system_configuration import \
    FileSystemConfiguration
from ..tuning.external_datasets import ExternalDatasets
from ..tuning.text_processor import TextProcessor
from .definition import Definition
from .compiler import Compiler
from .fitter import Fitter
from .evaluation import ModelEvaluation


class TestModelEvaluation(unittest.TestCase):
    """
    TestModelEvaluation class contains unt tests related to the
    ModelEvaluation class.
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

    def test_perform_regression_testing(self) -> None:
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

        ModelEvaluation.perform_regression_testing(
                test_model,
                data
            )

        self.assertTrue(
            len(ModelEvaluation.regression_test_records) != 0
        )

        ModelEvaluation.create_regression_test_results_csv()

        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    WoodgateSettings
                    .evaluation_summary_dir,
                    "regression_test_results.csv"
                )
            )
        )

        ModelEvaluation.create_confusion_matrix(
                test_model,
                data
            )

        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    WoodgateSettings
                    .evaluation_summary_dir,
                    "confusion_matrix.png"
                )
            )
        )

        test_report = \
            ModelEvaluation.create_classification_report(
                test_model,
                data
            )

        self.assertIsNotNone(test_report)

        train_acc, test_acc = \
            ModelEvaluation.evaluate_model_accuracy(
                test_model,
                data
            )

        self.assertEqual(type(train_acc), type(0.0))
        self.assertEqual(type(test_acc), type(0.0))

        ModelEvaluation\
            .create_regression_test_results_pie_chart()

        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    WoodgateSettings
                    .evaluation_summary_dir,
                    "regression_test_results.png"
                )
            )
        )


if __name__ == '__main__':
    unittest.main()
