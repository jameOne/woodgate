"""
file_system_configuration_test.py - Module -
The file_system_configuration_test.py module contains all unit
tests related to the file_system_configuration.py module.
"""
import os
import shutil
import unittest
from .file_system_configuration import FileSystemConfiguration
from ..woodgate_settings import WoodgateSettings


class TestFileSystemConfigurationDefaults(unittest.TestCase):
    """
    FileSystemConfigurationDefaultsTest - This class
    encapsulates all logic related to unit testing the
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

    def test_attrs(self):
        """

        :return:
        :rtype:
        """
        file_system_configuration = FileSystemConfiguration(
            woodgate_settings=WoodgateSettings
        )

        attrs = list()

        for attr, val in file_system_configuration\
                .__dict__.items():
            if attr[0] != "_":
                attrs.append(attr)

        self.assertEqual(len(attrs), 15)

    def test_default_values(self) -> None:
        """

        :return:
        :rtype:
        """
        file_system_configuration = FileSystemConfiguration(
            woodgate_settings=WoodgateSettings
        )

        exp_woodgate_base_dir = os.path.join(
                os.path.expanduser("~"),
                "woodgate"
            )

        # make sure woodgate base dir attr has a value.
        self.assertEqual(
            file_system_configuration.woodgate_base_dir,
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
            file_system_configuration.data_dir,
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
            file_system_configuration.output_dir,
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
            WoodgateSettings.build_version
        )

        # make sure build dir has a directory for data
        self.assertEqual(
            file_system_configuration.build_dir,
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
            WoodgateSettings.model_name
        )

        # make sure model build dir has a directory for the
        # specific model version
        self.assertEqual(
            file_system_configuration.model_build_dir,
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
            WoodgateSettings.model_name
        )

        # make sure model data dir has a directory for the
        # specific model version
        self.assertEqual(
            file_system_configuration.model_data_dir,
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
            WoodgateSettings.build_version
        )

        # make sure log dir has a directory for the
        # specific model version
        self.assertEqual(
            file_system_configuration.log_dir,
            exp_log_dir,
        )

        # make sure log directory is created if it does
        # not exist
        self.assertTrue(
            os.path.isdir(
                exp_log_dir
            )
        )

        exp_training_dir = os.path.join(
            exp_model_data_dir,
            "train"
        )

        # make sure training dir has a directory for the
        # specific model version
        self.assertEqual(
            file_system_configuration.training_dir,
            exp_training_dir,
        )

        # make sure training directory is created if it does
        # not exist
        self.assertTrue(
            os.path.isdir(
                exp_training_dir
            )
        )

        exp_testing_dir = os.path.join(
            exp_model_data_dir,
            "test"
        )

        # make sure testing dir has a directory for the
        # specific model version
        self.assertEqual(
            file_system_configuration.testing_dir,
            exp_testing_dir,
        )

        # make sure testing directory is created if it does
        # not exist
        self.assertTrue(
            os.path.isdir(
                exp_testing_dir
            )
        )

        exp_evaluation_dir = os.path.join(
            exp_model_data_dir,
            "evaluate"
        )

        # make sure evaluation dir has a directory for the
        # specific model version
        self.assertEqual(
            file_system_configuration.evaluation_dir,
            exp_evaluation_dir,
        )

        # make sure evaluation directory is created if it does
        # not exist
        self.assertTrue(
            os.path.isdir(
                exp_evaluation_dir
            )
        )

        exp_regression_dir = os.path.join(
            exp_model_data_dir,
            "regress"
        )

        # make sure regression dir has a directory for the
        # specific model version
        self.assertEqual(
            file_system_configuration.regression_dir,
            exp_regression_dir,
        )

        # make sure regression directory is created if it does
        # not exist
        self.assertTrue(
            os.path.isdir(
                exp_regression_dir
            )
        )

        exp_datasets_summary_dir = os.path.join(
            exp_build_dir,
            "datasets_summary"
        )

        # make sure there is a value for datasets summary
        # directory
        self.assertEqual(
            file_system_configuration.datasets_summary_dir,
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
            file_system_configuration.bert_dir,
            exp_bert_dir
        )

        # make sure bert directory is created if it
        # does not exist
        self.assertTrue(
            os.path.isdir(
                exp_bert_dir
            )
        )


if __name__ == '__main__':
    unittest.main()
