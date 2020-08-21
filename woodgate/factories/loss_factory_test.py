"""
loss_factory_test.py - The loss_factory_test.py module contains
unit tests related to the loss_factory.py module.
"""
import unittest
from .loss_factory import LossFactory
from tensorflow.keras import losses


class TestLossFactory(unittest.TestCase):
    """
    TestLossFactory class encapsulates unit tests related
    to the LossFactory class.
    """

    def test_get_loss(self) -> None:
        """

        :return:
        :rtype:
        """

        loss = LossFactory.get_loss(
            "Binary_Crossentropy",
            *["true", "0.5"]
        )

        self.assertTrue(
            isinstance(
                loss,
                losses.BinaryCrossentropy
            )
        )

        with self.assertRaises(ValueError):
            LossFactory.get_loss(
                "Binary_Crossentropy",
                *["true", "-0.5"]
            )

        loss = LossFactory.get_loss(
            "Categorical_Crossentropy",
            *["true", "0.5"]
        )

        self.assertTrue(
            isinstance(
                loss,
                losses.CategoricalCrossentropy
            )
        )

        with self.assertRaises(ValueError):
            LossFactory.get_loss(
                "Categorical_Crossentropy",
                *["true", "-0.5"]
            )

        loss = LossFactory.get_loss(
            "Categorical_Hinge",
            *["true", "0.5"]
        )

        self.assertTrue(
            isinstance(
                loss,
                losses.CategoricalHinge
            )
        )

        loss = LossFactory.get_loss(
            "Cosine_Similarity",
            *["1"]
        )

        self.assertTrue(
            isinstance(
                loss,
                losses.CosineSimilarity
            )
        )

        loss = LossFactory.get_loss(
            "Hinge"
        )

        self.assertTrue(
            isinstance(
                loss,
                losses.Hinge
            )
        )

        loss = LossFactory.get_loss(
            "Huber",
            *["1.0"]
        )

        self.assertTrue(
            isinstance(
                loss,
                losses.Huber
            )
        )

        loss = LossFactory.get_loss(
            "KL_Divergence",
            *["1.0"]
        )

        self.assertTrue(
            isinstance(
                loss,
                losses.KLDivergence
            )
        )

        loss = LossFactory.get_loss(
            "Log_Cosh",
            *["1.0"]
        )

        self.assertTrue(
            isinstance(
                loss,
                losses.LogCosh
            )
        )

        loss = LossFactory.get_loss(
            "Mean_Absolute_Error",
            *["1.0"]
        )

        self.assertTrue(
            isinstance(
                loss,
                losses.MeanAbsoluteError
            )
        )

        loss = LossFactory.get_loss(
            "Mean_Absolute_Error",
            *["1.0"]
        )

        self.assertTrue(
            isinstance(
                loss,
                losses.MeanAbsoluteError
            )
        )

        loss = LossFactory.get_loss(
            "Mean_Absolute_Percentage_Error",
            *["1.0"]
        )

        self.assertTrue(
            isinstance(
                loss,
                losses.MeanAbsolutePercentageError
            )
        )

        loss = LossFactory.get_loss(
            "Mean_Squared_Error",
            *["1.0"]
        )

        self.assertTrue(
            isinstance(
                loss,
                losses.MeanSquaredError
            )
        )

        loss = LossFactory.get_loss(
            "Mean_Squared_Logarithmic_Error",
            *["1.0"]
        )

        self.assertTrue(
            isinstance(
                loss,
                losses.MeanSquaredLogarithmicError
            )
        )

        loss = LossFactory.get_loss(
            "Poisson",
            *[]
        )

        self.assertTrue(
            isinstance(
                loss,
                losses.Poisson
            )
        )

        loss = LossFactory.get_loss(
            "Sparse_categorical_Crossentropy",
            *["True"]
        )

        self.assertTrue(
            isinstance(
                loss,
                losses.SparseCategoricalCrossentropy
            )
        )
        
        with self.assertRaises(ValueError):
            LossFactory.get_loss(
                "Invalid",
                *["True"]
            )


if __name__ == '__main__':
    unittest.main()
