"""
optimizer_factory_test.py - The optimizer_factory_test.py module
contains unit tests related to the optimizer_factory.py module.
"""
import unittest
from tensorflow.keras import optimizers
from .optimizer_factory import OptimizerFactory


class TestOptimizerFactory(unittest.TestCase):
    """
    TestOptimizerFactory class encapsulates unit tests related
    to the OptimizerFactory class.
    """

    def test_get_optimizer(self) -> None:
        """

        :return:
        :rtype:
        """
        optimizer = OptimizerFactory.get_optimizer(
            name="Adam",
            learning_rate=1e-5
        )

        self.assertTrue(
            isinstance(optimizer, optimizers.Adam)
        )

        optimizer = OptimizerFactory.get_optimizer(
            name="Adamax",
            learning_rate=1e-5
        )

        self.assertTrue(
            isinstance(optimizer, optimizers.Adamax)
        )

        optimizer = OptimizerFactory.get_optimizer(
            name="Adadelta",
            learning_rate=1e-5
        )

        self.assertTrue(
            isinstance(optimizer, optimizers.Adadelta)
        )

        optimizer = OptimizerFactory.get_optimizer(
            name="Adagrad",
            learning_rate=1e-5
        )

        self.assertTrue(
            isinstance(optimizer, optimizers.Adagrad)
        )

        optimizer = OptimizerFactory.get_optimizer(
            name="Ftrl",
            learning_rate=1e-5
        )

        self.assertTrue(
            isinstance(optimizer, optimizers.Ftrl)
        )

        optimizer = OptimizerFactory.get_optimizer(
            name="SGD",
            learning_rate=1e-5
        )

        self.assertTrue(
            isinstance(optimizer, optimizers.SGD)
        )

        optimizer = OptimizerFactory.get_optimizer(
            name="RMSprop",
            learning_rate=1e-5
        )

        self.assertTrue(
            isinstance(optimizer, optimizers.RMSprop)
        )

        with self.assertRaises(ValueError):
            OptimizerFactory.get_optimizer(
                name="invalid",
                learning_rate=1e-5
            )


if __name__ == '__main__':
    unittest.main()
