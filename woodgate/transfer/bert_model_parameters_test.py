"""
bert_model_test.py - This module contains the unit tests
related to the bert_model_test.py module.
"""
import os
import unittest
from .bert_model_parameters import BertModelParameters


class TestBertModelParametersDefault(unittest.TestCase):
    """
    TestBertModelParametersDefaults contains BertModel class
    unit tests when values are defaults.
    """

    def test_default_values(self) -> None:
        """

        :return:
        """
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
            BertModelParameters()

        self.assertRaises(
            ValueError,
            bert_model_parameters
        )




if __name__ == '__main__':
    unittest.main()

