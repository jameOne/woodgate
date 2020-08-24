"""
transfer_test.py - This module contains the unit
tests related to the bert_retrieval_strategy.py module.
"""
import os
import glob
import unittest
import shutil
from .bert_retrieval_strategy import BertRetrievalStrategy
from .bert_model_parameters import BertModelParameters
from ..woodgate_settings import Model, Build, FileSystem


class TestBertModelParametersDefault(unittest.TestCase):
    """
    TestBertModelParametersDefaults contains BertModel class
    unit tests when values are defaults.
    """

    def test_default_values(self) -> None:
        """

        :return:
        """
        os.environ["BERT_L_PARAM"] = "12"
        os.environ["BERT_H_PARAM"] = "768"
        os.environ["BERT_A_PARAM"] = "12"

        bert_model_parameters = BertModelParameters()

        # make sure there is a default L parameter
        self.assertEqual(
            bert_model_parameters.bert_l_param,
            12
        )

        # make sure there is a default H parameter
        self.assertEqual(
            bert_model_parameters.bert_h_param,
            768
        )

        # make sure there is a default A parameter
        self.assertEqual(
            bert_model_parameters.bert_a_param,
            12
        )

    def test_h_values_set_a_values(self) -> None:
        """

        :return:
        """
        os.environ["BERT_H_PARAM"] = "256"

        bert_model_parameters = BertModelParameters()

        self.assertEqual(
            bert_model_parameters.bert_a_param,
            4
        )

        os.environ["BERT_H_PARAM"] = "512"

        bert_model_parameters = BertModelParameters()

        self.assertEqual(
            bert_model_parameters.bert_a_param,
            8
        )

        os.environ["BERT_H_PARAM"] = "768"

        bert_model_parameters = BertModelParameters()

        self.assertEqual(
            bert_model_parameters.bert_a_param,
            12
        )

        os.environ["BERT_H_PARAM"] = "1024"

        bert_model_parameters = BertModelParameters()

        self.assertEqual(
            bert_model_parameters.bert_a_param,
            16
        )


class TestBertModelParameters(unittest.TestCase):
    """
    TestBertModelParameters tests BertModelParameters when
    setting non-default values.
    """

    def setUp(self) -> None:
        """

        :return:
        """
        os.environ["BERT_L_PARAM"] = "2"
        os.environ["BERT_H_PARAM"] = "128"
        os.environ["BERT_A_PARAM"] = "2"

    def test_values(self) -> None:
        """

        :return:
        """
        bert_model_parameters = BertModelParameters()

        # make sure there is an L parameter
        self.assertEqual(
            bert_model_parameters.bert_l_param,
            2
        )

        # make sure there is an H parameter
        self.assertEqual(
            bert_model_parameters.bert_h_param,
            128
        )

        # make sure there is an A parameter
        self.assertEqual(
            bert_model_parameters.bert_a_param,
            2
        )

    def test_l_not_allowed(self) -> None:
        """

        :return:
        """
        os.environ["BERT_L_PARAM"] = "0"

        def bert_model_parameters():
            """

            :return:
            :rtype:
            """
            BertModelParameters()

        self.assertRaises(
            ValueError,
            bert_model_parameters
        )

    def test_h_not_allowed(self) -> None:
        """

        :return:
        """
        os.environ["BERT_H_PARAM"] = "0"

        def bert_model_parameters():
            """

            :return:
            :rtype:
            """
            BertModelParameters()

        self.assertRaises(
            ValueError,
            bert_model_parameters
        )

    def test_init_w_params(self) -> None:
        """

        :return:
        :rtype:
        """
        bert_model_parameters = BertModelParameters(4, 256)

        self.assertEqual(
            bert_model_parameters.bert_l_param,
            4
        )

        self.assertEqual(
            bert_model_parameters.bert_h_param,
            256
        )

        self.assertEqual(
            bert_model_parameters.bert_a_param,
            4
        )


class TestBertRetrievalStrategy(unittest.TestCase):
    """
    TestBertRetrievalStrategy - The TestBertRetrievalStrategy
    tests whether the default BERT.
    """

    def setUp(self) -> None:
        """

        :return:
        """
        model = Model("test")
        build = Build()
        file_system = FileSystem(model, build)
        file_system.configure()
        self.file_system = file_system

    def tearDown(self) -> None:
        """

        :return:
        """
        shutil.rmtree(self.file_system.woodgate_base_dir)

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

        bert_retrieval_strategy.download_bert(self.file_system)

        self.assertTrue(
            glob.glob(
                f"{self.file_system.get_bert_model_path()}*"
            )
        )


if __name__ == '__main__':
    unittest.main()
