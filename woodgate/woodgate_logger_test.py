"""
woodgate_logger_test.py - The woodgate_logger_test.py module
contains the unit tests related to the woodgate_logger.py module.
"""
import unittest
import logging
from .woodgate_logger import WoodgateLogger


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
        woodgate_logger = WoodgateLogger.logger

        self.assertTrue(
            isinstance(woodgate_logger, logging.Logger)
        )


if __name__ == '__main__':
    unittest.main()
