"""
metrics_factory.py - The metrics_factory.py module contains the
MetricsFactory class definition.
"""
from typing import List
from tensorflow.keras import metrics


class MetricsFactory:
    """
    MetricsFactory class encapsulates logic related to
    generating the `metrics` to be used during the model
    compilation process.
    """

    @staticmethod
    def get_metrics(names: List[str]) -> List[metrics.Metric]:
        """

        :param names:
        :type names:
        :return:
        :rtype:
        """

        # ensure the name is lower case before
        # selecting the metrics_list.append(...)
        # statement
        names = [name.lower() for name in names]

        metrics_list = list()

        if "binary_crossentropy" in names:
            metrics_list.append(
                metrics.BinaryCrossentropy()
            )
        if "binary_accuracy" in names:
            metrics_list.append(
                metrics.BinaryAccuracy()
            )
        if "categorical_crossentropy" in names:
            metrics_list.append(
                metrics.CategoricalCrossentropy()
            )
        if "categorical_hinge" in names:
            metrics_list.append(
                metrics.CategoricalHinge()
            )
        if "cosine_similarity" in names:
            metrics_list.append(
                metrics.CosineSimilarity()
            )
        if "hinge" in names:
            metrics_list.append(
                metrics.Hinge()
            )
        if "kl_divergence" in names:
            metrics_list.append(
                metrics.KLDivergence()
            )
        if "mean_absolute_error" in names:
            metrics_list.append(
                metrics.MeanAbsoluteError()
            )
        if "mean_absolute_percentage_error" in names:
            metrics_list.append(
                metrics.MeanAbsolutePercentageError()
            )
        if "mean_squared_error" in names:
            metrics_list.append(
                metrics.MeanSquaredError()
            )
        if "mean_squared_logarithmic_error" in names:
            metrics_list.append(
                metrics.MeanSquaredLogarithmicError()
            )
        if "poisson" in names:
            metrics_list.append(
                metrics.Poisson()
            )
        if "sparse_categorical_crossentropy" in names:
            metrics_list.append(
                metrics.SparseCategoricalCrossentropy()
            )
        if "accuracy" in names:
            metrics_list.append(
                metrics.Accuracy()
            )
        if "auc" in names:
            metrics_list.append(
                metrics.AUC()
            )

        return metrics_list
