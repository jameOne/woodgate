"""
bert_retrieval_strategy_test.py - This module contains the unit
tests related to the bert_retrieval_strategy.py module.
"""
import os
import glob
import unittest
import shutil
from .bert_retrieval_strategy import BertRetrievalStrategy
from ..woodgate_settings import WoodgateSettings
from .bert_model_parameters import BertModelParameters


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

        bert_retrieval_strategy = BertRetrievalStrategy(
            bert_model_parameters=BertModelParameters(
                bert_h_param=128,
                bert_l_param=2
            )
        )

        bert_retrieval_strategy.download_bert()

        self.assertTrue(
            glob.glob(
                f"{WoodgateSettings.get_bert_model_path()}*"
            )
        )


if __name__ == '__main__':
    unittest.main()
