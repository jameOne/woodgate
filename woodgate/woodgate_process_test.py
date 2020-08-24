"""
woodgate_process_test.py - The
"""
import os
import unittest
from .woodgate_process import WoodgateProcess
from .woodgate_settings import WoodgateSettings
from .build_history.file_system_configuration import \
    FileSystemConfiguration


class TestWoodgateProcess(unittest.TestCase):
    """
    TestWoodgateProcess class encapsulates unit tests related to
    the WoodgateProcess class.
    """

    def test_run_w_visuals(self) -> None:
        """

        :return:
        :rtype:
        """
        FileSystemConfiguration(
            woodgate_settings=WoodgateSettings
        )

        WoodgateProcess.run()

        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    WoodgateSettings.evaluation_summary_dir,
                    "regression_test_results.csv"
                )
            )
        )

    def test_run_wo_visuals(self) -> None:
        """

        :return:
        :rtype:
        """
        WoodgateSettings.create_build_visuals = 0
        WoodgateSettings.create_dataset_visuals = 0
        WoodgateSettings.create_evaluation_visuals = 0

        FileSystemConfiguration(
            woodgate_settings=WoodgateSettings
        )

        WoodgateProcess.run()

        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    WoodgateSettings.evaluation_summary_dir,
                    "regression_test_results.csv"
                )
            )
        )

        # return settings to previous state for other tests
        WoodgateSettings.create_build_visuals = 1
        WoodgateSettings.create_dataset_visuals = 1
        WoodgateSettings.create_evaluation_visuals = 1



if __name__ == '__main__':
    unittest.main()
