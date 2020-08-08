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

from build_configuration import BuildConfiguration
from datasets import Datasets
from model_definition import ModelDefinition
from text_processor import TextProcessor


class ModelEvaluation:
    """
    ModelEvaluation - Class - The ModelEvaluation class encapsulates logic related to
    evaluating the model build.
    """

    @staticmethod
    def evaluate_model_accuracy(bert_model, data):
        """

        :param bert_model:
        :type bert_model:
        :param data:
        :type data:
        :return:
        :rtype:
        """
        # Model evaluation
        print("MODEL EVALUATION")
        _, train_acc = bert_model.evaluate(data.train_x, data.train_y)
        _, test_acc = bert_model.evaluate(data.test_x, data.test_y)

        print("train acc", train_acc)
        print("test acc", test_acc)

    @staticmethod
    def create_classification_report(bert_model, data):
        """

        :param bert_model:
        :type bert_model:
        :param data:
        :type data:
        :return:
        :rtype:
        """
        y_pred = bert_model.predict(data.test_x).argmax(axis=-1)
        print(classification_report(data.test_y, y_pred, target_names=Datasets.all_intents))

    @staticmethod
    def create_confusion_matrix(bert_model, data):
        """

        :param bert_model:
        :type bert_model:
        :param data:
        :type data:
        :return:
        :rtype:
        """
        y_pred = bert_model.predict(data.test_x).argmax(axis=-1)
        print(classification_report(data.test_y, y_pred, target_names=Datasets.all_intents))
        # Confusion matrix
        cm = confusion_matrix(data.test_y, y_pred)
        df_cm = pd.DataFrame(cm, index=Datasets.all_intents, columns=Datasets.all_intents)

        heat_map = sns.heatmap(df_cm, annot=True, fmt="d")
        heat_map.yaxis.set_ticklabels(heat_map.yaxis.get_ticklabels(), rotation=0, ha='right')
        heat_map.xaxis.set_ticklabels(heat_map.xaxis.get_ticklabels(), rotation=30, ha='right')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('Confusion matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(
            BuildConfiguration.EVALUATION_DIR, "confusion_matrix.png"))
        plt.figure().clear()

    @staticmethod
    def perform_regression_testing(bert_model, data):
        """

        :return:
        :rtype:
        """
        pred_tokens = map(ModelDefinition.tokenizer.tokenize, Datasets.regression_data[TextProcessor.DATA_COLUMN])
        pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
        pred_token_ids = list(map(ModelDefinition.tokenizer.convert_tokens_to_ids, pred_tokens))

        pred_token_ids = map(
            lambda token_ids: token_ids + [0] * (data.max_sequence_length - len(token_ids)), pred_token_ids)
        pred_token_ids = np.array(list(pred_token_ids))

        predictions = bert_model.predict(pred_token_ids).argmax(axis=-1)

        for utterance, intent in zip(Datasets.regression_data[TextProcessor.DATA_COLUMN], predictions):
            print("utterance:", utterance, "\nintent:", Datasets.all_intents[intent])
