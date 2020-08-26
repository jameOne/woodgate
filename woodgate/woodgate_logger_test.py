"""
woodgate_logger_test.py - The woodgate_logger_test.py module
contains the unit tests related to the woodgate_logger.py module.
"""
import unittest
import logging
from .woodgate_logger import WoodgateLogger
from .woodgate_settings import Model, Build, FileSystem


class TestWoodgateLogger(unittest.TestCase):
    """
    TestWoodgateLogger contains the unit tests related to the
    WoodgateLogger class.
    """

    def test_logger(self) -> None:
        """

        :return:
        :rtype:
        """
        model = Model("test")
        build = Build()
        file_system = FileSystem(model, build)
        file_system.configure()

        woodgate_logger = WoodgateLogger(file_system=file_system)

        self.assertTrue(
            isinstance(woodgate_logger.logger, logging.Logger)
        )


if __name__ == '__main__':
    unittest.main()
