"""
evaluator.py - Module - This module contains the Evaluator
class which encapsulates logic related to evaluating the evaluator
build_history.
"""
import os
import json
from typing import Tuple, Any, Dict, List
import numpy as np
from tensorflow import keras
from woodgate.woodgate_settings import FileSystem
from woodgate.tuning.external_datasets import \
    ExternalDatasets
from woodgate.trainer.preprocessor import Preprocessor


class Evaluator:
    """
    Evaluator - The Evaluator class encapsulates logic related to
    evaluating the evaluator build_history.
    """

    #: The `regression_test_records` attribute represents a list
    #:
    regression_test_records: List[Dict[str, Any]] = list()

    @staticmethod
    def evaluate_model_accuracy(
            model: keras.Model,
            data: Preprocessor
    ) -> Tuple[Any, Any]:
        """This method wraps calls which evaluate the evaluator
        on the provided data.

        :param model: The application specific (trained) \
        BERT evaluator.
        :type model: keras.Model
        :param data: Processed textual data.
        :type data: Preprocessor
        :return: A tuple of the training accuracy, and testing \
        accuracy respectively.
        :rtype: Tuple[Any, Any]
        """
        train = model.evaluate(
            data.train_x,
            data.train_y
        )
        test = model.evaluate(
            data.test_x,
            data.test_y
        )

        return train, test

    @classmethod
    def perform_regression_testing(
            cls,
            model: keras.Model,
            data: Preprocessor,
            file_system: FileSystem
    ) -> None:
        """This method will perform regression testing on the
        evaluator (it is assumed this method is called after
        training). Where regression testing differs from the
        other tests in that the result is recorded and a report
        is generated which considers successive evaluator builds
        for a time series representation of the evaluator's
        accuracy over the complete build_history history.

        :param model:
        :type model:
        :param data:
        :type data:
        :param file_system:
        :type file_system:
        :return:
        :rtype:
        """

        # TODO - Deliver on the doc string.
        vocab_file = file_system.get_bert_vocab_path()
        pred_tokens = map(
            Preprocessor.tokenizer_factory(vocab_file).tokenize,
            ExternalDatasets.regression_data[
                Preprocessor.data_column_title
            ]
        )
        pred_tokens = map(
            lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
        pred_token_ids = map(
                Preprocessor.tokenizer_factory(vocab_file)
                .convert_tokens_to_ids,
                pred_tokens
            )

        pred_token_ids = map(
            lambda token_ids: token_ids + [0] * (
                    data.max_sequence_length - len(token_ids)
            ),
            pred_token_ids
        )
        pred_token_ids = np.array(list(pred_token_ids))

        predictions = model.predict(pred_token_ids).argmax(
            axis=-1)

        regression_test_records = list()
        for text, expected_label, label in zip(
                ExternalDatasets.regression_data[
                    Preprocessor.data_column_title
                ],
                ExternalDatasets.regression_data[
                    Preprocessor.label_column_title
                ],
                predictions
        ):
            actual_label = ExternalDatasets.all_intents()[
                label
            ]
            match = expected_label == actual_label
            regression_test_records.append(
                {
                    "text": text,
                    "expected_label": expected_label,
                    "actual_label": actual_label,
                    "match": match
                }
            )

        cls.regression_test_records = regression_test_records

        return None

    @classmethod
    def create_regression_test_results_json(
            cls,
            file_system: FileSystem
    ) -> None:
        """

        :return:
        :rtype:
        """

        regression_test_results = {
            "results": cls.regression_test_records
        }

        regression_test_results_json_path = os.path.join(
            file_system.evaluation_summary_dir,
            "regressionTestResults.json"
        )

        with open(
                regression_test_results_json_path,
                "w+"
        ) as file:
            file.write(json.dumps(regression_test_results))

        return None
