"""
metrics_factory_test.py - The metrics_factory_test.py module
contains all unit tests related to the metrics_factory module.
"""
import unittest
from tensorflow.keras import metrics
from .metrics_factory import MetricsFactory


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

        metrics_list = MetricsFactory.get_metrics(names=names)

        self.assertTrue(
            isinstance(
                metrics_list[0],
                metrics.BinaryCrossentropy
            )
        )
        self.assertTrue(
            isinstance(
                metrics_list[1],
                metrics.BinaryAccuracy
            )
        )
        self.assertTrue(
            isinstance(
                metrics_list[2],
                metrics.CategoricalCrossentropy
            )
        )
        self.assertTrue(
            isinstance(
                metrics_list[3],
                metrics.CategoricalHinge
            )
        )
        self.assertTrue(
            isinstance(
                metrics_list[4],
                metrics.CosineSimilarity
            )
        )
        self.assertTrue(
            isinstance(
                metrics_list[5],
                metrics.Hinge
            )
        )
        self.assertTrue(
            isinstance(
                metrics_list[6],
                metrics.KLDivergence
            )
        )
        self.assertTrue(
            isinstance(
                metrics_list[7],
                metrics.MeanAbsoluteError
            )
        )
        self.assertTrue(
            isinstance(
                metrics_list[8],
                metrics.MeanAbsolutePercentageError
            )
        )
        self.assertTrue(
            isinstance(
                metrics_list[9],
                metrics.MeanSquaredError
            )
        )
        self.assertTrue(
            isinstance(
                metrics_list[10],
                metrics.MeanSquaredLogarithmicError
            )
        )
        self.assertTrue(
            isinstance(
                metrics_list[11],
                metrics.Poisson
            )
        )
        self.assertTrue(
            isinstance(
                metrics_list[12],
                metrics.SparseCategoricalCrossentropy
            )
        )
        self.assertTrue(
            isinstance(
                metrics_list[13],
                metrics.Accuracy
            )
        )
        self.assertTrue(
            isinstance(
                metrics_list[14],
                metrics.AUC
            )
        )

        names = []
        metrics_list = MetricsFactory.get_metrics(
            names=names
        )
        self.assertTrue(len(metrics_list) == 0)


if __name__ == '__main__':
    unittest.main()
