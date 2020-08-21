"""
compiler_test.py - The compiler_test.py module contains unit
tests related to the compiler.py module.
"""
import os
import unittest
from tensorflow import keras
from .compiler import Compiler


class TestCompiler(unittest.TestCase):
    """
    TestCompiler class encapsulates the unit tests related to
    the Compiler class.
    """

    def test_default_values(self) -> None:
        """

        :return:
        :rtype:
        """

        self.assertIsNone(Compiler.compile(keras.Model()))


if __name__ == '__main__':
    unittest.main()
