"""
woodgate_process_test.py - The
"""
import os
import unittest
from .woodgate_process import WoodgateProcess
from .woodgate_settings import Model, Build, FileSystem


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
        model = Model("test")
        build = Build()
        file_system = FileSystem(model, build)
        file_system.configure()

        WoodgateProcess.run(model=model, file_system=file_system)

        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    file_system.evaluation_summary_dir,
                    "regressionTestResults.json"
                )
            )
        )


if __name__ == '__main__':
    unittest.main()
