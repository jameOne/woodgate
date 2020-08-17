"""
evaluation.py - Module - This module contains the ModelEvaluation
class which encapsulates logic related to evaluating the model
build.
"""
import os
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)
import pandas as pd
import seaborn as sns
import numpy as np
from typing import Tuple, Any, Dict
import matplotlib.pyplot as plt
from tensorflow import keras

from ..tuning.external_datasets import \
    ExternalDatasets
from ..model.definition import Definition
from ..tuning.text_processor import TextProcessor
from ..build.file_system_configuration import \
    FileSystemConfiguration


class ModelEvaluation:
    """
    ModelEvaluation - Class - The ModelEvaluation class
    encapsulates logic related to evaluating the model build.
    """

    @staticmethod
    def evaluate_model_accuracy(
            bert_model: keras.Model,
            data: TextProcessor
    ) -> Tuple[Any, Any]:
        """This method wraps calls which evaluate the model on the
        provided data.

        :param bert_model: The application specific (trained) \
        BERT model.
        :type bert_model: keras.Model
        :param data: Processed textual data.
        :type data: TextProcessor
        :return: A tuple of the training accuracy, and testing \
        accuracy respectively.
        :rtype: Tuple[Any, Any]
        """
        _, train_acc = bert_model.evaluate(
            data.train_x,
            data.train_y
        )
        _, test_acc = bert_model.evaluate(
            data.test_x,
            data.test_y
        )

        return train_acc, test_acc

    @staticmethod
    def create_classification_report(
            bert_model: keras.Model,
            data: TextProcessor
    ) -> Dict:
        """This method generates a report describing on the
        model's ability to classify the textual data.

        :param bert_model: The application specific (trained) \
        BERT model.
        :type bert_model: keras.Model
        :param data: Processed textual data.
        :type data: TextProcessor
        :return: A dictionary containing classification data.
        :rtype: Dict
        """
        y_pred = bert_model.predict(data.test_x).argmax(axis=-1)
        report_dict = classification_report(
                data.test_y,
                y_pred,
                target_names=ExternalDatasets.all_intents()
            )

        return report_dict

    @staticmethod
    def create_confusion_matrix(
            bert_model: keras.Model,
            data: TextProcessor
    ) -> None:
        """This model will generate a confusion matrix from the
        trained model and processed textual data.

        :param bert_model: The application specific (trained) \
        BERT model.
        :type bert_model: keras.Model
        :param data: Processed textual data.
        :type data: TextProcessor
        :return: None
        :rtype: NoneType
        """
        y_pred = bert_model.predict(data.test_x).argmax(axis=-1)
        print(
            classification_report(
                data.test_y,
                y_pred,
                target_names=ExternalDatasets.all_intents()
            )
        )
        # Confusion matrix
        cm = confusion_matrix(data.test_y, y_pred)
        df_cm = pd.DataFrame(
            cm,
            index=ExternalDatasets.all_intents(),
            columns=ExternalDatasets.all_intents()
        )

        heat_map = sns.heatmap(df_cm, annot=True, fmt="d")
        heat_map.yaxis.set_ticklabels(
            heat_map.yaxis.get_ticklabels(),
            rotation=0,
            ha='right'
        )
        heat_map.xaxis.set_ticklabels(
            heat_map.xaxis.get_ticklabels(),
            rotation=30,
            ha='right'
        )
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('Confusion matrix')
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                FileSystemConfiguration.evaluation_dir,
                "confusion_matrix.png"
            )
        )
        plt.figure().clear()

        return None

    @staticmethod
    def perform_regression_testing(
            bert_model: keras.Model,
            data: TextProcessor
    ) -> None:
        """This method will perform regression testing on the
        model (it is assumed this method is called after training)
        . Where regression testing differs from the other tests in
        that the result is recorded and a report is generated
        which considers successive model builds for a time series
        representation of the model's accuracy over the complete
        build history.

        :param bert_model: The application specific (trained) \
        BERT model.
        :type bert_model: keras.Model
        :param data: Processed textual data.
        :type data: TextProcessor
        :return: None
        :rtype: NoneType
        """

        # TODO - Deliver on the doc string.
        pred_tokens = map(
            Definition.get_tokenizer().tokenize,
            ExternalDatasets.regression_data[
                TextProcessor.data_column_title
            ]
        )
        pred_tokens = map(
            lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
        pred_token_ids = list(
            map(
                Definition.get_tokenizer().convert_tokens_to_ids,
                pred_tokens
            )
        )

        pred_token_ids = map(
            lambda token_ids: token_ids
            + [0] * (data.max_sequence_length - len(token_ids)),
            pred_token_ids
        )
        pred_token_ids = np.array(list(pred_token_ids))

        predictions = bert_model.predict(pred_token_ids).argmax(
            axis=-1)

        for utterance, intent in zip(
                ExternalDatasets.regression_data[
                    TextProcessor.data_column_title
                ],
                predictions
        ):
            print("utterance:", utterance, "\nintent:",
                  ExternalDatasets.all_intents()[intent])
