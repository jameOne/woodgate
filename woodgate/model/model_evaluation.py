"""
model_evaluation.py - Module - This module contains the ModelEvaluation class which encapsulates logic related to
evaluating the model build.
"""
import os
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

from ..build.build_configuration import BuildConfiguration
from ..fine_tuning.fine_tuning_datasets import FineTuningDatasets
from ..model.model_definition import ModelDefinition
from ..fine_tuning.fine_tuning_text_processor import FineTuningTextProcessor


class ModelEvaluation:
    """
    ModelEvaluation - Class - The ModelEvaluation class encapsulates logic related to
    evaluating the model build.
    """

    def __init__(
            self,
            build_configuration: BuildConfiguration,
            model_definition: ModelDefinition,
            fine_tuning_datasets: FineTuningDatasets,
    ):
        self.evaluation_dir = build_configuration.evaluation_dir
        self.tokenizer = model_definition.tokenizer
        self.all_intents = fine_tuning_datasets.all_intents
        self.regression_data = fine_tuning_datasets.regression_data

    @staticmethod
    def evaluate_model_accuracy(
            bert_model: keras.Model,
            data: FineTuningTextProcessor
    ):
        """

        :param bert_model:
        :param data:
        :return:
        """
        # Model evaluation
        print("MODEL EVALUATION")
        _, train_acc = bert_model.evaluate(data.train_x, data.train_y)
        _, test_acc = bert_model.evaluate(data.test_x, data.test_y)

        print("train acc", train_acc)
        print("test acc", test_acc)

    def create_classification_report(
            self,
            bert_model: keras.Model,
            data: FineTuningTextProcessor
    ):
        """

        :param bert_model:
        :param data:
        :return:
        """
        y_pred = bert_model.predict(data.test_x).argmax(axis=-1)
        print(classification_report(data.test_y, y_pred, target_names=self.all_intents))

    def create_confusion_matrix(self, bert_model, data):
        """

        :param bert_model:
        :param data:
        :return:
        """
        y_pred = bert_model.predict(data.test_x).argmax(axis=-1)
        print(classification_report(data.test_y, y_pred, target_names=self.all_intents))
        # Confusion matrix
        cm = confusion_matrix(data.test_y, y_pred)
        df_cm = pd.DataFrame(cm, index=self.all_intents, columns=self.all_intents)

        heat_map = sns.heatmap(df_cm, annot=True, fmt="d")
        heat_map.yaxis.set_ticklabels(heat_map.yaxis.get_ticklabels(), rotation=0, ha='right')
        heat_map.xaxis.set_ticklabels(heat_map.xaxis.get_ticklabels(), rotation=30, ha='right')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('Confusion matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(
            self.evaluation_dir, "confusion_matrix.png"))
        plt.figure().clear()

    def perform_regression_testing(self, bert_model, data: FineTuningTextProcessor):
        """

        :return:
        :rtype:
        """
        pred_tokens = map(self.tokenizer.tokenize, self.regression_data[FineTuningTextProcessor.DATA_COLUMN])
        pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
        pred_token_ids = list(map(self.tokenizer.convert_tokens_to_ids, pred_tokens))

        pred_token_ids = map(
            lambda token_ids: token_ids + [0] * (data.max_sequence_length - len(token_ids)), pred_token_ids)
        pred_token_ids = np.array(list(pred_token_ids))

        predictions = bert_model.predict(pred_token_ids).argmax(axis=-1)

        for utterance, intent in zip(self.regression_data[FineTuningTextProcessor.DATA_COLUMN], predictions):
            print("utterance:", utterance, "\nintent:", self.all_intents[intent])
