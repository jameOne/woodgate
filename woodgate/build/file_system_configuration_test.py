"""
file_system_configuration_test.py - Module -
The file_system_configuration_test.py module contains all unit
tests related to the file_system_configuration.py module.
"""
import os
import shutil
import datetime
from uuid import UUID
import unittest
from .file_system_configuration import FileSystemConfiguration


class TestFileSystemConfigurationDefaults(unittest.TestCase):
    """
    FileSystemConfigurationDefaultsTest - This class encapsulates
    all logic related to unit testing the
    FileSystemConfiguration class using default file
    system configuration.
    """
    def setUp(self) -> None:
        """

        :return:
        :rtype:
        """
        pass

    def tearDown(self) -> None:
        """

        :return:
        :rtype:
        """
        woodgate_base_dir = os.path.join(
            os.path.expanduser("~"),
            "woodgate"
        )
        shutil.rmtree(woodgate_base_dir)

    def test_default_values(self) -> None:
        """

        :return:
        :rtype:
        """
        # the following will throw if the model_name is not a
        # valid UUID which is virtually equivalent to an
        # assertion.
        UUID(FileSystemConfiguration.model_name, version=4)
        self.assertTrue(
            type(
                datetime.datetime.strptime(
                    FileSystemConfiguration.build_version,
                    "%Y_%m_%d-%H:%M:%S"
                )
            ),
            type(datetime.datetime.now())
        )

        # make sure create dataset visuals attr has a value
        self.assertEqual(
            FileSystemConfiguration.create_dataset_visuals,
            1
        )

        # make sure create build visuals attr has a value
        self.assertEqual(
            FileSystemConfiguration.create_build_visuals,
            1
        )

        exp_woodgate_base_dir = os.path.join(
                os.path.expanduser("~"),
                "woodgate"
            )

        # make sure woodgate base dir attr has a value.
        self.assertEqual(
            FileSystemConfiguration.woodgate_base_dir,
            exp_woodgate_base_dir
        )

        # make sure woodgate base directory is created if it does
        # not exist
        self.assertTrue(
            os.path.isdir(
                exp_woodgate_base_dir
            )
        )

        exp_data_dir = os.path.join(
            exp_woodgate_base_dir,
            "data"
        )

        # make sure woodgate base dir has a directory for data
        self.assertEqual(
            FileSystemConfiguration.data_dir,
            exp_data_dir
        )

        # make sure data directory is created if it does not
        # exist
        self.assertTrue(
            os.path.isdir(
                exp_data_dir
            )
        )

        exp_output_dir = os.path.join(
            exp_woodgate_base_dir,
            "output"
        )

        # make sure output dir has a directory for data
        self.assertEqual(
            FileSystemConfiguration.output_dir,
            exp_output_dir
        )

        # make sure output directory is created if it does not
        # exist
        self.assertTrue(
            os.path.isdir(
                exp_output_dir
            )
        )

        exp_build_dir = os.path.join(
            exp_output_dir,
            FileSystemConfiguration.build_version
        )

        # make sure build dir has a directory for data
        self.assertEqual(
            FileSystemConfiguration.build_dir,
            exp_build_dir
        )

        # make sure build directory is created if it does not
        # exist
        self.assertTrue(
            os.path.isdir(
                exp_build_dir
            )
        )

        exp_model_build_dir = os.path.join(
            exp_build_dir,
            FileSystemConfiguration.model_name
        )

        # make sure model build dir has a directory for the
        # specific model version
        self.assertEqual(
            FileSystemConfiguration.model_build_dir,
            exp_model_build_dir,
        )

        # make sure model build directory is created if it does
        # not exist
        self.assertTrue(
            os.path.isdir(
                exp_model_build_dir
            )
        )

        exp_model_data_dir = os.path.join(
            exp_data_dir,
            FileSystemConfiguration.model_name
        )

        # make sure model data dir has a directory for the
        # specific model version
        self.assertEqual(
            FileSystemConfiguration.model_data_dir,
            exp_model_data_dir,
        )

        # make sure model data directory is created if it does
        # not exist
        self.assertTrue(
            os.path.isdir(
                exp_model_data_dir
            )
        )

        exp_log_dir = os.path.join(
            exp_output_dir,
            "log",
            FileSystemConfiguration.build_version
        )

        # make sure log dir has a directory for the
        # specific model version
        self.assertEqual(
            FileSystemConfiguration.log_dir,
            exp_log_dir,
        )

        # make sure log directory is created if it does
        # not exist
        self.assertTrue(
            os.path.isdir(
                exp_log_dir
            )
        )

        exp_log_file = \
            f"{FileSystemConfiguration.build_version}.log"

        # make sure there is a value for log file
        self.assertEqual(
            FileSystemConfiguration.log_file,
            exp_log_file
        )

        # make sure there is a value for log path
        self.assertEqual(
            FileSystemConfiguration.log_path,
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
            FileSystemConfiguration.training_dir,
            exp_training_dir,
        )

        # make sure training directory is created if it does
        # not exist
        self.assertTrue(
            os.path.isdir(
                exp_training_dir
            )
        )

        exp_training_file = "train.csv"

        # make sure there is a value for training file
        self.assertEqual(
            FileSystemConfiguration.training_file,
            exp_training_file
        )

        # make sure there is a value for training path
        self.assertEqual(
            FileSystemConfiguration.training_path,
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
            FileSystemConfiguration.testing_dir,
            exp_testing_dir,
        )

        # make sure testing directory is created if it does
        # not exist
        self.assertTrue(
            os.path.isdir(
                exp_testing_dir
            )
        )

        exp_testing_file = "test.csv"

        # make sure there is a value for testing file
        self.assertEqual(
            FileSystemConfiguration.testing_file,
            exp_testing_file
        )

        # make sure there is a value for testing path
        self.assertEqual(
            FileSystemConfiguration.testing_path,
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
            FileSystemConfiguration.evaluation_dir,
            exp_evaluation_dir,
        )

        # make sure evaluation directory is created if it does
        # not exist
        self.assertTrue(
            os.path.isdir(
                exp_evaluation_dir
            )
        )

        exp_evaluation_file = "evaluate.csv"

        # make sure there is a value for evaluation file
        self.assertEqual(
            FileSystemConfiguration.evaluation_file,
            exp_evaluation_file
        )

        # make sure there is a value for evaluation path
        self.assertEqual(
            FileSystemConfiguration.evaluation_path,
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
            FileSystemConfiguration.regression_dir,
            exp_regression_dir,
        )

        # make sure regression directory is created if it does
        # not exist
        self.assertTrue(
            os.path.isdir(
                exp_regression_dir
            )
        )

        exp_regression_file = "regress.csv"

        # make sure there is a value for regression file
        self.assertEqual(
            FileSystemConfiguration.regression_file,
            exp_regression_file
        )

        # make sure there is a value for regression path
        self.assertEqual(
            FileSystemConfiguration.regression_path,
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
            FileSystemConfiguration.datasets_summary_dir,
            exp_datasets_summary_dir
        )

        # make sure datasets summary directory is created if it
        # does not exist
        self.assertTrue(
            os.path.isdir(
                exp_datasets_summary_dir
            )
        )

        exp_bert_dir = os.path.join(
            exp_woodgate_base_dir,
            "bert"
        )

        # make sure there is a value for bert
        # directory
        self.assertEqual(
            FileSystemConfiguration.bert_dir,
            exp_bert_dir
        )

        # make sure bert directory is created if it
        # does not exist
        self.assertTrue(
            os.path.isdir(
                exp_bert_dir
            )
        )

        exp_bert_config_file = "bert_config.json"

        # make sure there is a value for bert config
        self.assertEqual(
            FileSystemConfiguration.bert_config_file,
            exp_bert_config_file
        )

        # make sure there is a value for bert config path
        self.assertEqual(
            FileSystemConfiguration.bert_config_path,
            os.path.join(
                exp_bert_dir,
                exp_bert_config_file
            )
        )

        exp_bert_model_file = "bert_model.ckpt"

        # make sure there is a value for bert model
        self.assertEqual(
            FileSystemConfiguration.bert_model_file,
            exp_bert_model_file
        )

        # make sure there is a value for bert model path
        self.assertEqual(
            FileSystemConfiguration.bert_model_path,
            os.path.join(
                exp_bert_dir,
                exp_bert_model_file
            )
        )

        exp_bert_vocab_file = "vocab.txt"

        # make sure there is a value for bert vocab file
        self.assertEqual(
            FileSystemConfiguration.bert_vocab_file,
            exp_bert_vocab_file
        )

        # make sure there is a value for bert vocab path
        self.assertEqual(
            FileSystemConfiguration.bert_vocab_path,
            os.path.join(
                exp_bert_dir,
                exp_bert_vocab_file
            )
        )


if __name__ == '__main__':
    unittest.main()
