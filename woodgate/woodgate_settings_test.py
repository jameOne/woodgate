"""
woodgate_settings_test.py - The woodgate_settings_test.py module
contains unit tests related to the woodgate_setting.py module.
"""
import os
import datetime
from uuid import UUID
import unittest
from .woodgate_settings import WoodgateSettings


class TestWoodgateSettingsDefaults(unittest.TestCase):
    """
    FileSystemConfigurationDefaultsTest - This class encapsulates
    all logic related to unit testing the
    FileSystemConfiguration class using default file
    system configuration.
    """

    def test_default_values(self) -> None:
        """

        :return:
        :rtype:
        """
        # the following will throw if the model_name is not a
        # valid UUID which is virtually equivalent to an
        # assertion.
        UUID(WoodgateSettings.model_name, version=4)
        self.assertTrue(
            type(
                datetime.datetime.strptime(
                    WoodgateSettings.build_version,
                    "%Y_%m_%d-%H_%M_%S"
                )
            ),
            type(datetime.datetime.now())
        )

        # make sure create dataset visuals attr has a value
        self.assertEqual(
            WoodgateSettings.create_dataset_visuals,
            1
        )

        # make sure create build visuals attr has a value
        self.assertEqual(
            WoodgateSettings.create_build_visuals,
            1
        )

        exp_woodgate_base_dir = os.path.join(
                os.path.expanduser("~"),
                "woodgate"
            )

        # make sure woodgate base dir attr has a value.
        self.assertEqual(
            WoodgateSettings.woodgate_base_dir,
            exp_woodgate_base_dir
        )

        exp_data_dir = os.path.join(
            exp_woodgate_base_dir,
            "data"
        )

        # make sure woodgate base dir has a directory for data
        self.assertEqual(
            WoodgateSettings.data_dir,
            exp_data_dir
        )

        exp_output_dir = os.path.join(
            exp_woodgate_base_dir,
            "output"
        )

        # make sure output dir has a directory for data
        self.assertEqual(
            WoodgateSettings.output_dir,
            exp_output_dir
        )

        exp_build_dir = os.path.join(
            exp_output_dir,
            WoodgateSettings.build_version
        )

        # make sure build dir has a directory for data
        self.assertEqual(
            WoodgateSettings.build_dir,
            exp_build_dir
        )

        exp_model_build_dir = os.path.join(
            exp_build_dir,
            WoodgateSettings.model_name
        )

        # make sure model build dir has a directory for the
        # specific model version
        self.assertEqual(
            WoodgateSettings.model_build_dir,
            exp_model_build_dir,
        )

        exp_model_data_dir = os.path.join(
            exp_data_dir,
            WoodgateSettings.model_name
        )

        # make sure model data dir has a directory for the
        # specific model version
        self.assertEqual(
            WoodgateSettings.model_data_dir,
            exp_model_data_dir,
        )

        exp_log_dir = os.path.join(
            exp_output_dir,
            "log",
            WoodgateSettings.build_version
        )

        # make sure log dir has a directory for the
        # specific model version
        self.assertEqual(
            WoodgateSettings.log_dir,
            exp_log_dir,
        )

        exp_log_file = \
            f"{WoodgateSettings.build_version}.log"

        # make sure there is a value for log file
        self.assertEqual(
            WoodgateSettings.log_file,
            exp_log_file
        )

        # make sure there is a value for log path
        self.assertEqual(
            WoodgateSettings.get_log_path(),
            os.path.join(
                exp_log_dir,
                exp_log_file
            )
        )

        exp_training_dir = os.path.join(
            exp_model_data_dir,
            "train"
        )

        # make sure training dir has a directory for the
        # specific model version
        self.assertEqual(
            WoodgateSettings.training_dir,
            exp_training_dir,
        )

        exp_training_file = "train.csv"

        # make sure there is a value for training file
        self.assertEqual(
            WoodgateSettings.training_file,
            exp_training_file
        )

        # make sure there is a value for training path
        self.assertEqual(
            WoodgateSettings.get_training_path(),
            os.path.join(
                exp_training_dir,
                exp_training_file
            )
        )

        exp_testing_dir = os.path.join(
            exp_model_data_dir,
            "test"
        )

        # make sure testing dir has a directory for the
        # specific model version
        self.assertEqual(
            WoodgateSettings.testing_dir,
            exp_testing_dir,
        )

        exp_testing_file = "test.csv"

        # make sure there is a value for testing file
        self.assertEqual(
            WoodgateSettings.testing_file,
            exp_testing_file
        )

        # make sure there is a value for testing path
        self.assertEqual(
            WoodgateSettings.get_testing_path(),
            os.path.join(
                exp_testing_dir,
                exp_testing_file
            )
        )

        exp_evaluation_dir = os.path.join(
            exp_model_data_dir,
            "evaluate"
        )

        # make sure evaluation dir has a directory for the
        # specific model version
        self.assertEqual(
            WoodgateSettings.evaluation_dir,
            exp_evaluation_dir,
        )

        exp_evaluation_file = "evaluate.csv"

        # make sure there is a value for evaluation file
        self.assertEqual(
            WoodgateSettings.evaluation_file,
            exp_evaluation_file
        )

        # make sure there is a value for evaluation path
        self.assertEqual(
            WoodgateSettings.get_evaluation_path(),
            os.path.join(
                exp_evaluation_dir,
                exp_evaluation_file
            )
        )

        exp_regression_dir = os.path.join(
            exp_model_data_dir,
            "regress"
        )

        # make sure regression dir has a directory for the
        # specific model version
        self.assertEqual(
            WoodgateSettings.regression_dir,
            exp_regression_dir,
        )

        exp_regression_file = "regress.csv"

        # make sure there is a value for regression file
        self.assertEqual(
            WoodgateSettings.regression_file,
            exp_regression_file
        )

        # make sure there is a value for regression path
        self.assertEqual(
            WoodgateSettings.get_regression_path(),
            os.path.join(
                exp_regression_dir,
                exp_regression_file
            )
        )

        exp_datasets_summary_dir = os.path.join(
            exp_build_dir,
            "datasets_summary"
        )

        # make sure there is a value for datasets summary
        # directory
        self.assertEqual(
            WoodgateSettings.datasets_summary_dir,
            exp_datasets_summary_dir
        )

        exp_bert_dir = os.path.join(
            exp_woodgate_base_dir,
            "bert"
        )

        # make sure there is a value for bert
        # directory
        self.assertEqual(
            WoodgateSettings.bert_dir,
            exp_bert_dir
        )

        exp_bert_config_file = "bert_config.json"

        # make sure there is a value for bert config
        self.assertEqual(
            WoodgateSettings.bert_config_file,
            exp_bert_config_file
        )

        # make sure there is a value for bert config path
        self.assertEqual(
            WoodgateSettings.get_bert_config_path(),
            os.path.join(
                exp_bert_dir,
                exp_bert_config_file
            )
        )

        exp_bert_model_file = \
            "bert_model.ckpt"

        # make sure there is a value for bert model
        self.assertEqual(
            WoodgateSettings.bert_model_file,
            exp_bert_model_file
        )

        # make sure there is a value for bert model path
        self.assertEqual(
            WoodgateSettings.get_bert_model_path(),
            os.path.join(
                exp_bert_dir,
                exp_bert_model_file
            )
        )

        exp_bert_vocab_file = "vocab.txt"

        # make sure there is a value for bert vocab file
        self.assertEqual(
            WoodgateSettings.bert_vocab_file,
            exp_bert_vocab_file
        )

        # make sure there is a value for bert vocab path
        self.assertEqual(
            WoodgateSettings.get_bert_vocab_path(),
            os.path.join(
                exp_bert_dir,
                exp_bert_vocab_file
            )
        )


if __name__ == '__main__':
    unittest.main()
