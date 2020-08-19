"""
bert_retrieval_strategy_test.py - This module contains the unit
tests related to the bert_retrieval_strategy.py module.
"""
import os
import unittest
import shutil
from .bert_retrieval_strategy import BertRetrievalStrategy
from ..build.file_system_configuration import \
    FileSystemConfiguration


class TestBertRetrievalStrategy(unittest.TestCase):
    """
    TestBertRetrievalStrategy - The TestBertRetrievalStrategy
    tests whether the default BERT.
    """

    def setUp(self) -> None:
        """

        :return:
        """
        pass

    def tearDown(self) -> None:
        """

        :return:
        """
        woodgate_base_dir = os.path.join(
            os.path.expanduser("~"),
            "woodgate"
        )
        shutil.rmtree(woodgate_base_dir)

    def test_default_bert_retrieval(self) -> None:
        """

        :return:
        """
        BertRetrievalStrategy.download_bert()

        self.assertTrue(
            os.path.isfile(
                FileSystemConfiguration.bert_model_path
            )
        )


if __name__ == '__main__':
    unittest.main()
