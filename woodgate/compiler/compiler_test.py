"""
compiler_test.py - The compiler_test.py module contains unit
tests related to the compiler.py module.
"""
import unittest
from tensorflow import keras
from .compiler import Compiler


class TestCompile(unittest.TestCase):
    """
    TestCompiler class encapsulates the unit tests related to
    the Compiler class.
    """

    def test_default_values(self) -> None:
        """

        :return:
        :rtype:
        """

        optimizer = Compiler.optimizer_factory(
            name="Adam",
            learning_rate=1e-5
        )

        loss = Compiler.loss_factory(
            "Binary_Crossentropy",
            *["true", "0.5"]
        )

        names = [
            "binary_crossentropy",
        ]

        metrics = Compiler.metrics_factory(*names)

        self.assertIsNone(
            Compiler.compile(
                model=keras.Model(),
                optimizer=optimizer,
                loss=loss,
                metrics=metrics
            )
        )


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
        optimizer = Compiler.optimizer_factory(
            name="Adam",
            learning_rate=1e-5
        )

        self.assertTrue(
            isinstance(optimizer, keras.optimizers.Adam)
        )

        optimizer = Compiler.optimizer_factory(
            name="Adamax",
            learning_rate=1e-5
        )

        self.assertTrue(
            isinstance(optimizer, keras.optimizers.Adamax)
        )

        optimizer = Compiler.optimizer_factory(
            name="Adadelta",
            learning_rate=1e-5
        )

        self.assertTrue(
            isinstance(optimizer, keras.optimizers.Adadelta)
        )

        optimizer = Compiler.optimizer_factory(
            name="Adagrad",
            learning_rate=1e-5
        )

        self.assertTrue(
            isinstance(optimizer, keras.optimizers.Adagrad)
        )

        optimizer = Compiler.optimizer_factory(
            name="Ftrl",
            learning_rate=1e-5
        )

        self.assertTrue(
            isinstance(optimizer, keras.optimizers.Ftrl)
        )

        optimizer = Compiler.optimizer_factory(
            name="SGD",
            learning_rate=1e-5
        )

        self.assertTrue(
            isinstance(optimizer, keras.optimizers.SGD)
        )

        optimizer = Compiler.optimizer_factory(
            name="RMSprop",
            learning_rate=1e-5
        )

        self.assertTrue(
            isinstance(optimizer, keras.optimizers.RMSprop)
        )

        with self.assertRaises(ValueError):
            Compiler.optimizer_factory(
                name="invalid",
                learning_rate=1e-5
            )


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

        loss = Compiler.loss_factory(
            "Binary_Crossentropy",
            *["true", "0.5"]
        )

        self.assertTrue(
            isinstance(
                loss,
                keras.losses.BinaryCrossentropy
            )
        )

        with self.assertRaises(ValueError):
            Compiler.loss_factory(
                "Binary_Crossentropy",
                *["true", "-0.5"]
            )

        loss = Compiler.loss_factory(
            "Categorical_Crossentropy",
            *["true", "0.5"]
        )

        self.assertTrue(
            isinstance(
                loss,
                keras.losses.CategoricalCrossentropy
            )
        )

        with self.assertRaises(ValueError):
            Compiler.loss_factory(
                "Categorical_Crossentropy",
                *["true", "-0.5"]
            )

        loss = Compiler.loss_factory(
            "Categorical_Hinge",
            *["true", "0.5"]
        )

        self.assertTrue(
            isinstance(
                loss,
                keras.losses.CategoricalHinge
            )
        )

        loss = Compiler.loss_factory(
            "Cosine_Similarity",
            *["1"]
        )

        self.assertTrue(
            isinstance(
                loss,
                keras.losses.CosineSimilarity
            )
        )

        loss = Compiler.loss_factory(
            "Hinge"
        )

        self.assertTrue(
            isinstance(
                loss,
                keras.losses.Hinge
            )
        )

        loss = Compiler.loss_factory(
            "Huber",
            *["1.0"]
        )

        self.assertTrue(
            isinstance(
                loss,
                keras.losses.Huber
            )
        )

        loss = Compiler.loss_factory(
            "KL_Divergence",
            *["1.0"]
        )

        self.assertTrue(
            isinstance(
                loss,
                keras.losses.KLDivergence
            )
        )

        loss = Compiler.loss_factory(
            "Log_Cosh",
            *["1.0"]
        )

        self.assertTrue(
            isinstance(
                loss,
                keras.losses.LogCosh
            )
        )

        loss = Compiler.loss_factory(
            "Mean_Absolute_Error",
            *["1.0"]
        )

        self.assertTrue(
            isinstance(
                loss,
                keras.losses.MeanAbsoluteError
            )
        )

        loss = Compiler.loss_factory(
            "Mean_Absolute_Error",
            *["1.0"]
        )

        self.assertTrue(
            isinstance(
                loss,
                keras.losses.MeanAbsoluteError
            )
        )

        loss = Compiler.loss_factory(
            "Mean_Absolute_Percentage_Error",
            *["1.0"]
        )

        self.assertTrue(
            isinstance(
                loss,
                keras.losses.MeanAbsolutePercentageError
            )
        )

        loss = Compiler.loss_factory(
            "Mean_Squared_Error",
            *["1.0"]
        )

        self.assertTrue(
            isinstance(
                loss,
                keras.losses.MeanSquaredError
            )
        )

        loss = Compiler.loss_factory(
            "Mean_Squared_Logarithmic_Error",
            *["1.0"]
        )

        self.assertTrue(
            isinstance(
                loss,
                keras.losses.MeanSquaredLogarithmicError
            )
        )

        loss = Compiler.loss_factory(
            "Poisson",
            *[]
        )

        self.assertTrue(
            isinstance(
                loss,
                keras.losses.Poisson
            )
        )

        loss = Compiler.loss_factory(
            "Sparse_categorical_Crossentropy",
            *["True"]
        )

        self.assertTrue(
            isinstance(
                loss,
                keras.losses.SparseCategoricalCrossentropy
            )
        )

        with self.assertRaises(ValueError):
            Compiler.loss_factory(
                "Invalid",
                *["True"]
            )


class TestMetricsFactory(unittest.TestCase):
    """
    TestMetricsFactory class contains the unit tests relates to
    the MetricsFactory class.
    """

    def test_get_metrics(self) -> None:
        """

        :return:
        :rtype:
        """
        names = [
            "binary_crossentropy",
            "binary_accuracy",
            "categorical_crossentropy",
            "categorical_hinge",
            "cosine_similarity",
            "hinge",
            "kl_divergence",
            "mean_absolute_error",
            "mean_absolute_percentage_error",
            "mean_squared_error",
            "mean_squared_logarithmic_error",
            "poisson",
            "sparse_categorical_crossentropy",
            "accuracy",
            "auc"
        ]

        metrics_list = Compiler.metrics_factory(*names)

        self.assertTrue(
            isinstance(
                metrics_list[0],
                keras.metrics.BinaryCrossentropy
            )
        )
        self.assertTrue(
            isinstance(
                metrics_list[1],
                keras.metrics.BinaryAccuracy
            )
        )
        self.assertTrue(
            isinstance(
                metrics_list[2],
                keras.metrics.CategoricalCrossentropy
            )
        )
        self.assertTrue(
            isinstance(
                metrics_list[3],
                keras.metrics.CategoricalHinge
            )
        )
        self.assertTrue(
            isinstance(
                metrics_list[4],
                keras.metrics.CosineSimilarity
            )
        )
        self.assertTrue(
            isinstance(
                metrics_list[5],
                keras.metrics.Hinge
            )
        )
        self.assertTrue(
            isinstance(
                metrics_list[6],
                keras.metrics.KLDivergence
            )
        )
        self.assertTrue(
            isinstance(
                metrics_list[7],
                keras.metrics.MeanAbsoluteError
            )
        )
        self.assertTrue(
            isinstance(
                metrics_list[8],
                keras.metrics.MeanAbsolutePercentageError
            )
        )
        self.assertTrue(
            isinstance(
                metrics_list[9],
                keras.metrics.MeanSquaredError
            )
        )
        self.assertTrue(
            isinstance(
                metrics_list[10],
                keras.metrics.MeanSquaredLogarithmicError
            )
        )
        self.assertTrue(
            isinstance(
                metrics_list[11],
                keras.metrics.Poisson
            )
        )
        self.assertTrue(
            isinstance(
                metrics_list[12],
                keras.metrics.SparseCategoricalCrossentropy
            )
        )
        self.assertTrue(
            isinstance(
                metrics_list[13],
                keras.metrics.Accuracy
            )
        )
        self.assertTrue(
            isinstance(
                metrics_list[14],
                keras.metrics.AUC
            )
        )

        names = []
        metrics_list = Compiler.metrics_factory(
            *names
        )
        self.assertTrue(len(metrics_list) == 0)


if __name__ == '__main__':
    unittest.main()
