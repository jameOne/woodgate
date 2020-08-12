"""
fine_tuning_datasets.py - This module contains the Dataset class definition which encapsulates logic related to
training, testing, and validation datasets.
"""
import os
import json
import pandas as pd
from ..build.build_configuration import BuildConfiguration
from matplotlib_venn import venn2
import matplotlib.pyplot as plt


class FineTuningDatasets:
    """
    Dataset - Class - The Dataset class encapsulates logic related to
    training, testing, and validation datasets.
    """

    def __init__(self, build_configuration: BuildConfiguration):
        """

        :param build_configuration:
        """
        #: The `training_dir` attribute represents the path to a directory on the host file system where the
        #: training data will be stored. The training data directory should be a child directory of the model data
        #: directory in which the program will store the data retrieved for training the learning model. This attribute
        #: is set via the `TRAINING_DIR` environment variable. If the `TRAINING_DIR` environment variable is not
        #: set, then the `training_dir` attribute will default to `$MODEL_DATA_DIR/train`. The program will attempt to
        #: create `TRAINING_DIR` if it does not already exist.
        self.training_dir: str = os.getenv("TRAINING_DIR", os.path.join(build_configuration.model_data_dir, "train"))
        os.makedirs(self.training_dir, exist_ok=True)

        #: The `training_file` attribute represents the base name of the `training_path` attribute. The training file
        #: should therefore reside in the `training_dir` by definition. This file should have a CSV file (having a
        #: `.csv` file extension). This attribute is set via the `TRAINING_FILE` environment variable. The training
        #: file should contain at least two (2) columns; one (1) for text having title `text_column_title`
        #: (set via `TEXT_COLUMN_TITLE` environment variable) and one (1) for label having title `label_column_title`
        #: (set via `LABEL_COLUMN_TITLE` environment variable).
        self.training_file: str = os.getenv("TRAINING_FILE", "train.csv")

        #: The `training_path` attribute represents the full path on the host file system pointing to `training_file`.
        #: This attribute is set via the `TRAINING_PATH` environment variable. If `TRAINING_PATH` is set, then it will
        #: render values set by `training_dir` and `training_file` inconsequential. If `TRAINING_PATH` is not set, then
        #: the `training_path` will default to `$TRAINING_DIR/$TRAINING_FILE`.
        self.training_path: str = os.getenv("TRAINING_PATH", os.path.join(self.training_dir, self.training_file))

        #: The `testing_dir` attribute represents the path to a directory on the host file system where the
        #: testing data will be stored. The testing data directory should be a child directory of the model data
        #: directory in which the program will store the data retrieved for testing the learning model. This attribute
        #: is set via the `TESTING_DIR` environment variable. If the `TESTING_DIR` environment variable is not
        #: set, then the `testing_dir` attribute will default to `$MODEL_DATA_DIR/test`. The program will attempt to
        #: create `TESTING_DIR` if it does not already exist.
        self.testing_dir: str = os.getenv("TESTING_DIR", os.path.join(build_configuration.model_data_dir, "test"))
        os.makedirs(self.testing_dir, exist_ok=True)

        #: The `testing_file` attribute represents the base name of the `testing_path` attribute. The testing file
        #: should therefore reside in the `testing_dir` by definition. This file should have a CSV file (having a
        #: `.csv` file extension). This attribute is set via the `TESTING_FILE` environment variable. The testing
        #: file should contain at least two (2) columns; one (1) for text having title `text_column_title`
        #: (set via `TEXT_COLUMN_TITLE` environment variable) and one (1) for label having title `label_column_title`
        #: (set via `LABEL_COLUMN_TITLE` environment variable).
        self.testing_file: str = os.getenv("TESTING_FILE", "test.csv")

        #: The `testing_path` attribute represents the full path on the host file system pointing to `testing_file`.
        #: This attribute is set via the `TESTING_PATH` environment variable. If `TESTING_PATH` is set, then it will
        #: render values set by `testing_dir` and `testing_file` inconsequential. If `TESTING_PATH` is not set, then
        #: the `testing_path` will default to `$TESTING_DIR/$TESTING_FILE`.
        self.testing_path: str = os.path.join(self.testing_dir, self.testing_file)

        #: The `validation_dir` attribute represents the path to a directory on the host file system where the
        #: validation data will be stored. The validation data directory should be a child directory of the model data
        #: directory in which the program will store the data retrieved for validating the learning model. This
        #: attribute is set via the `VALIDATION_DIR` environment variable. If the `VALIDATION_DIR` environment variable
        #: is not set, then the `validation_dir` attribute will default to `$MODEL_DATA_DIR/validate`. The program will
        #: attempt to create `VALIDATION_DIR` if it does not already exist.
        self.validation_dir: str = os.getenv("VALIDATION_DIR",
                                             os.path.join(build_configuration.model_data_dir, "validate"))
        os.makedirs(self.validation_dir, exist_ok=True)

        #: The `validation_file` attribute represents the base name of the `validation_path` attribute. The validation
        #: file should therefore reside in the `validation_dir` by definition. This file should have a CSV file (having
        #: a `.csv` file extension). This attribute is set via the `VALIDATION_FILE` environment variable. The
        #: validation file should contain at least two (2) columns; one (1) for text having title `text_column_title`
        #: (set via `TEXT_COLUMN_TITLE` environment variable) and one (1) for label having title `label_column_title`
        #: (set via `LABEL_COLUMN_TITLE` environment variable).
        self.validation_file: str = os.getenv("VALIDATION_FILE", "validate.csv")

        #: The `validation_path` attribute represents the full path on the host file system pointing to
        #: `validation_file`. This attribute is set via the `VALIDATION_PATH` environment variable. If
        #: `VALIDATION_PATH` is set, then it will render values set by `validation_dir` and `validation_file`
        #: inconsequential. If  `VALIDATION_PATH` is not set, then the `validation_path` will default to
        #: `$VALIDATION_DIR/$VALIDATION_FILE`.
        self.validation_path: str = os.path.join(self.validation_dir, self.validation_file)

        #: The `evaluation_dir` attribute represents the path to a directory on the host file system where the
        #: evaluation data will be stored. The evaluation data directory should be a child directory of the model data
        #: directory in which the program will store the data retrieved for validating the learning model. This
        #: attribute is set via the `EVALUATION_DIR` environment variable. If the `EVALUATION_DIR` environment variable
        #: is not set, then the `evaluation_dir` attribute will default to `$MODEL_DATA_DIR/evaluate`. The program will
        #: attempt to create `EVALUATION_DIR` if it does not already exist.
        self.evaluation_dir: str = os.getenv("EVALUATION_DIR",
                                             os.path.join(build_configuration.model_data_dir, "evaluate"))
        os.makedirs(self.evaluation_dir, exist_ok=True)

        #: The `evaluation_file` attribute represents the base name of the `evaluation_path` attribute. The evaluation
        #: file should therefore reside in the `evaluation_dir` by definition. This file should have a CSV file (having
        #: a `.csv` file extension). This attribute is set via the `EVALUATION_FILE` environment variable. The
        #: evaluation file should contain at least two (2) columns; one (1) for text having title `text_column_title`
        #: (set via `TEXT_COLUMN_TITLE` environment variable) and one (1) for label having title `label_column_title`
        #: (set via `LABEL_COLUMN_TITLE` environment variable).
        self.evaluation_file: str = os.getenv("EVALUATION_FILE", "evaluate.csv")

        #: The `evaluation_path` attribute represents the full path on the host file system pointing to
        #: `evaluation_file`. This attribute is set via the `EVALUATION_PATH` environment variable. If
        #: `EVALUATION_PATH` is set, then it will render values set by `evaluation_dir` and `evaluation_file`
        #: inconsequential. If  `EVALUATION_PATH` is not set, then the `evaluation_path` will default to
        #: `$EVALUATION_DIR/$EVALUATION_FILE`.
        self.evaluation_path: str = os.path.join(self.evaluation_dir, self.evaluation_file)

        #: The `regression_dir` attribute represents the path to a directory on the host file system where the
        #: regression data will be stored. The regression data directory should be a child directory of the model data
        #: directory in which the program will store the data retrieved for validating the learning model. This
        #: attribute is set via the `REGRESSION_DIR` environment variable. If the `REGRESSION_DIR` environment variable
        #: is not set, then the `regression_dir` attribute will default to `$MODEL_DATA_DIR/regress`. The program will
        #: attempt to create `REGRESSION_DIR` if it does not already exist.
        self.regression_dir: str = os.getenv("REGRESSION_DIR",
                                             os.path.join(build_configuration.model_data_dir, "regress"))
        os.makedirs(self.regression_dir, exist_ok=True)

        #: The `regression_file` attribute represents the base name of the `regression_path` attribute. The regression
        #: file should therefore reside in the `regression_dir` by definition. This file should have a CSV file (having
        #: a `.csv` file extension). This attribute is set via the `REGRESSION_FILE` environment variable. The
        #: regression file should contain at least two (2) columns; one (1) for text having title `text_column_title`
        #: (set via `TEXT_COLUMN_TITLE` environment variable) and one (1) for label having title `label_column_title`
        #: (set via `LABEL_COLUMN_TITLE` environment variable).
        self.regression_file: str = os.getenv("REGRESSION_FILE", "regress.csv")

        #: The `regression_path` attribute represents the full path on the host file system pointing to
        #: `regression_file`. This attribute is set via the `REGRESSION_PATH` environment variable. If
        #: `REGRESSION_PATH` is set, then it will render values set by `regression_dir` and `regression_file`
        #: inconsequential. If  `REGRESSION_PATH` is not set, then the `regression_path` will default to
        #: `$REGRESSION_DIR/$REGRESSION_FILE`.
        self.regression_path: str = os.path.join(self.regression_dir, self.regression_file)

        # Read training data into dataframe
        self.training_data = pd.read_csv(self.training_path)
        self.training_intents_list = self.training_data.intent.unique().tolist()
        self.training_intents_set = set(self.training_intents_list)
        self.training_intents_counts = self.training_data.intent.value_counts()

        # Read testing data into dataframe
        self.testing_data = pd.read_csv(self.testing_path)
        self.testing_intents_list = self.testing_data.intent.unique().tolist()
        self.testing_intents_set = set(self.testing_intents_list)
        self.testing_intents_counts = self.testing_data.intent.value_counts()

        # Read validation data into dataframe
        self.validation_data = pd.read_csv(self.validation_path)
        self.validation_intents_list = self.validation_data.intent.unique().tolist()
        self.validation_intents_set = set(self.validation_intents_list)
        self.validation_intents_counts = self.validation_data.intent.value_counts()

        # Read evaluation data into dataframe
        self.evaluation_data = pd.read_csv(self.evaluation_path)
        self.evaluation_intents_list = self.evaluation_data.intent.unique().tolist()
        self.evaluation_intents_set = set(self.evaluation_intents_list)
        self.evaluation_intents_counts = self.evaluation_data.intent.value_counts()

        # Read regression data into dataframe
        self.regression_data = pd.read_csv(self.regression_path)
        self.regression_intents_list = self.regression_data.intent.unique().tolist()
        self.regression_intents_set = set(self.regression_intents_list)
        self.regression_intents_counts = self.regression_data.intent.value_counts()

        # TODO - There needs to be a re-definition for all intents. Basically
        #  it should be the training intent list + warnings and errors for
        #  cases people try to test, validate, or regress on intent sets that
        #  are larger than the set of training intents.
        self.all_intents = list(set(self.training_intents_list + self.testing_intents_list
                                    + self.validation_intents_list + self.validation_intents_list))

        self.data_set_summary_dir = os.getenv("DATASET_SUMMARY_DIR",
                                              os.path.join(build_configuration.build_dir, "dataset_summary"))

        self.intents_data = {
            "intents": self.all_intents,
            "testing_set": {
                "intent": self.testing_intents_list,
                "count": self.testing_intents_counts
            },
            "training_set": {
                "intent": self.training_intents_list,
                "count": self.training_intents_counts
            },
            "validation_set": {
                "intent": self.validation_intents_list,
                "count": self.validation_intents_counts
            }
        }

    def create_intents_data_json(self):
        """

        :return:
        :rtype:
        """
        intents_data_json = os.path.join(self.data_set_summary_dir, "intentsData.json")
        with open(intents_data_json) as file:
            file.write(json.dumps(self.intents_data))

    def create_intents_bar_plots(self):
        """

        :return:
        :rtype:
        """
        fig, axs = plt.subplots()
        axs.title.set_text("Testing set")
        axs.barh(self.testing_intents_list, self.testing_intents_counts)
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_set_summary_dir, "intents_bar_plot_testing.png"))
        plt.figure().clear()

        fig, axs = plt.subplots()
        axs.title.set_text("Training set")
        axs.barh(self.training_intents_list, self.training_intents_counts)
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_set_summary_dir, "intents_bar_plot_training.png"))
        plt.figure().clear()

        fig, axs = plt.subplots()
        axs.title.set_text("Validation set")
        axs.barh(self.validation_intents_list, self.validation_intents_counts)
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_set_summary_dir, "intents_bar_plot_validation.png"))
        plt.figure().clear()

        fig, axs = plt.subplots()
        axs.title.set_text("Regression set")
        axs.barh(self.regression_intents_list, self.regression_intents_counts)
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_set_summary_dir, "intents_bar_plot_regression.png"))
        plt.figure().clear()

    def create_intents_venn_diagrams(self):
        """

        :return:
        :rtype:
        """

        plt.figure(figsize=(4, 4))
        venn2(
            [
                self.training_intents_set,
                self.validation_intents_set,
            ],
            ("Training", "Validation")
        )
        plt.title("Datasets - Intents Venn Diagram - Training and Validation")
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_set_summary_dir, "intents_venn_training_validation.png"))
        plt.figure().clear()

        plt.figure(figsize=(4, 4))
        venn2(
            [
                self.training_intents_set,
                self.testing_intents_set,
            ],
            ("Training", "Testing")
        )
        plt.title("Datasets - Intents Venn Diagram - Training and Testing")
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_set_summary_dir, "intents_venn_training_testing.png"))
        plt.figure().clear()

        plt.figure(figsize=(4, 4))
        venn2(
            [
                self.training_intents_set,
                self.regression_intents_set,
            ],
            ("Training", "Regression")
        )
        plt.title("Datasets - Intents Venn Diagram - Training and Regression")
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_set_summary_dir, "intents_venn_training_regression.png"))
        plt.figure().clear()

        plt.figure(figsize=(4, 4))
        venn2(
            [
                self.validation_intents_set,
                self.testing_intents_set,
            ],
            ("Validation", "Testing")
        )
        plt.title("Datasets - Intents Venn Diagram - Validation and Testing")
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_set_summary_dir, "intents_venn_validation_testing.png"))
        plt.figure().clear()

        plt.figure(figsize=(4, 4))
        venn2(
            [
                self.validation_intents_set,
                self.regression_intents_set,
            ],
            ("Validation", "Regression")
        )
        plt.title("Datasets - Intents Venn Diagram - Validation and Regression")
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_set_summary_dir, "intents_venn_validation_regression.png"))
        plt.figure().clear()

        plt.figure(figsize=(4, 4))
        venn2(
            [
                self.testing_intents_set,
                self.regression_intents_set,
            ],
            ("Testing", "Regression")
        )
        plt.title("Datasets - Intents Venn Diagram - Testing and Regression")
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_set_summary_dir, "intents_venn_testing_regression.png"))
        plt.figure().clear()
