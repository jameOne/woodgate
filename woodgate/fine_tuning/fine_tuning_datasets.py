"""
fine_tuning_datasets.py - The fine_tuning_datasets.py module
contains the FineTuningDataset class definition.
"""
import os
import json
from typing import List, Set, Dict, Union
import pandas as pd
from ..build.build_configuration import BuildConfiguration
from matplotlib_venn import venn2
import matplotlib.pyplot as plt
from ..woodgate_logger import WoodgateLogger


class FineTuningDatasets:
    """
    FineTuningDatasets - The Dataset class encapsulates logic
    related to training, testing, evaluation, and regression
    datasets.
    """
    training_intents_counts: object

    def __init__(
            self,
            build_configuration: BuildConfiguration
    ):
        """

        :param build_configuration:
        """
        #: The `training_dir` attribute represents the path to a
        #: directory on the host file system where the
        #: training data will be stored. The training data
        #: directory should be a child directory of the model data
        #: directory in which the program will store the data
        #: retrieved for training the learning model. This
        #: attribute is set via the `TRAINING_DIR` environment
        #: variable. If the `TRAINING_DIR` environment variable
        #: is not set, then the `training_dir` attribute will
        #: default to `$MODEL_DATA_DIR/train`. The program will
        #: attempt to create `TRAINING_DIR` if it does not
        #: already exist.
        self.training_dir: str = os.getenv(
            "TRAINING_DIR",
            os.path.join(
                build_configuration.model_data_dir,
                "train"
            )
        )
        os.makedirs(self.training_dir, exist_ok=True)

        #: The `training_file` attribute represents the base
        #: name of the `training_path` attribute. The training
        #: file should therefore reside in the `training_dir` by
        #: definition. This file should have a CSV file (having a
        #: `.csv` file extension). This attribute is set via the
        #: `TRAINING_FILE` environment variable. The training
        #: file should contain at least two (2) columns; one (1)
        #: for text having title `text_column_title` (set via
        #: `TEXT_COLUMN_TITLE` environment variable) and one (1)
        #: for label having title `label_column_title`
        #: (set via `LABEL_COLUMN_TITLE` environment variable).
        self.training_file: str = os.getenv(
            "TRAINING_FILE",
            "train.csv"
        )

        #: The `training_path` attribute represents the full path
        #: on the host file system pointing to `training_file`.
        #: This attribute is set via the `TRAINING_PATH`
        #: environment variable. If `TRAINING_PATH` is set, then
        #: it will render values set by `training_dir` and
        #: `training_file` inconsequential. If `TRAINING_PATH` is
        #: not set, then the `training_path` will default to
        #: `$TRAINING_DIR/$TRAINING_FILE`.
        self.training_path: str = os.getenv(
            "TRAINING_PATH",
            os.path.join(
                self.training_dir,
                self.training_file
            )
        )

        #: The `testing_dir` attribute represents the path to a
        #: directory on the host file system where the
        #: testing data will be stored. The testing data directory
        #: should be a child directory of the model data
        #: directory in which the program will store the data
        #: retrieved for testing the learning model. This
        #: attribute is set via the `TESTING_DIR` environment
        #: variable. If the `TESTING_DIR` environment variable is
        #: not set, then the `testing_dir` attribute will default
        #: to `$MODEL_DATA_DIR/test`. The program will attempt to
        #: create `TESTING_DIR` if it does not already exist.
        self.testing_dir: str = os.getenv(
            "TESTING_DIR",
            os.path.join(
                build_configuration.model_data_dir,
                "test"
            )
        )
        os.makedirs(self.testing_dir, exist_ok=True)

        #: The `testing_file` attribute represents the base name
        #: of the `testing_path` attribute. The testing file
        #: should therefore reside in the `testing_dir` by
        #: definition. This file should have a CSV file (having a
        #: `.csv` file extension). This attribute is set via the
        #: `TESTING_FILE` environment variable. The testing
        #: file should contain at least two (2) columns; one (1)
        #: for text having title `text_column_title`
        #: (set via `TEXT_COLUMN_TITLE` environment variable) and
        #: one (1) for label having title `label_column_title`
        #: (set via `LABEL_COLUMN_TITLE` environment variable).
        self.testing_file: str = os.getenv(
            "TESTING_FILE",
            "test.csv"
        )

        #: The `testing_path` attribute represents the full path
        #: on the host file system pointing to `testing_file`.
        #: This attribute is set via the `TESTING_PATH`
        #: environment variable. If `TESTING_PATH` is set, then it
        #: will render values set by `testing_dir` and
        #: `testing_file` inconsequential. If `TESTING_PATH` is
        #: not set, then the `testing_path` will default to
        #: `$TESTING_DIR/$TESTING_FILE`.
        self.testing_path: str = os.path.join(
            self.testing_dir,
            self.testing_file
        )

        #: The `evaluation_dir` attribute represents the path to
        #: a directory on the host file system where the
        #: evaluation data will be stored. The evaluation data
        #: directory should be a child directory of the model data
        #: directory in which the program will store the data
        #: retrieved for validating the learning model. This
        #: attribute is set via the `EVALUATION_DIR` environment
        #: variable. If the `EVALUATION_DIR` environment variable
        #: is not set, then the `evaluation_dir` attribute will
        #: default to `$MODEL_DATA_DIR/evaluate`. The program will
        #: attempt to create `EVALUATION_DIR` if it does not
        #: already exist.
        self.evaluation_dir: str = os.getenv(
            "EVALUATION_DIR",
            os.path.join(
                build_configuration.model_data_dir,
                "evaluate"
            )
        )
        os.makedirs(self.evaluation_dir, exist_ok=True)

        #: The `evaluation_file` attribute represents the base
        #: name of the `evaluation_path` attribute. The evaluation
        #: file should therefore reside in the `evaluation_dir`
        #: by definition. This file should have a CSV file (having
        #: a `.csv` file extension). This attribute is set via
        #: the `EVALUATION_FILE` environment variable. The
        #: evaluation file should contain at least two (2)
        #: columns; one (1) for text having title
        #: `text_column_title` (set via `TEXT_COLUMN_TITLE`
        #: environment variable) and one (1) for label having
        #: title `label_column_title` (set via
        #: `LABEL_COLUMN_TITLE` environment variable).
        self.evaluation_file: str = os.getenv(
            "EVALUATION_FILE",
            "evaluate.csv"
        )

        #: The `evaluation_path` attribute represents the full
        #: path on the host file system pointing to
        #: `evaluation_file`. This attribute is set via the
        #: `EVALUATION_PATH` environment variable. If
        #: `EVALUATION_PATH` is set, then it will render values
        #: set by `evaluation_dir` and `evaluation_file`
        #: inconsequential. If  `EVALUATION_PATH` is not set,
        #: then the `evaluation_path` will default to
        #: `$EVALUATION_DIR/$EVALUATION_FILE`.
        self.evaluation_path: str = os.path.join(
            self.evaluation_dir,
            self.evaluation_file
        )

        #: The `regression_dir` attribute represents the path to
        #: a directory on the host file system where the
        #: regression data will be stored. The regression data
        #: directory should be a child directory of the model data
        #: directory in which the program will store the data
        #: retrieved for validating the learning model. This
        #: attribute is set via the `REGRESSION_DIR` environment
        #: variable. If the `REGRESSION_DIR` environment variable
        #: is not set, then the `regression_dir` attribute will
        #: default to `$MODEL_DATA_DIR/regress`. The program will
        #: attempt to create `REGRESSION_DIR` if it does not
        #: already exist.
        self.regression_dir: str = os.getenv(
            "REGRESSION_DIR",
            os.path.join(
                build_configuration.model_data_dir,
                "regress"
            )
        )
        os.makedirs(self.regression_dir, exist_ok=True)

        #: The `regression_file` attribute represents the base
        #: name of the `regression_path` attribute. The regression
        #: file should therefore reside in the `regression_dir`
        #: by definition. This file should have a CSV file (having
        #: a `.csv` file extension). This attribute is set via
        #: the `REGRESSION_FILE` environment variable. The
        #: regression file should contain at least two (2) columns
        #: ; one (1) for text having title `text_column_title`
        #: (set via `TEXT_COLUMN_TITLE` environment variable)
        #: and one (1) for label having title `label_column_title`
        #: (set via `LABEL_COLUMN_TITLE` environment variable).
        self.regression_file: str = os.getenv(
            "REGRESSION_FILE",
            "regress.csv"
        )

        #: The `regression_path` attribute represents the full
        #: path on the host file system pointing to
        #: `regression_file`. This attribute is set via the
        #: `REGRESSION_PATH` environment variable. If
        #: `REGRESSION_PATH` is set, then it will render values
        #: set by `regression_dir` and `regression_file`
        #: inconsequential. If  `REGRESSION_PATH` is not set,
        #: then the `regression_path` will default to
        #: `$REGRESSION_DIR/$REGRESSION_FILE`.
        self.regression_path: str = os.path.join(
            self.regression_dir,
            self.regression_file
        )

        #: The `training_data` attribute represents the fine
        #: tuning data designated as "training data". This dataset
        #: is used to tune the hyper-parameters during the model
        #: training iterations. This dataset should be a CSV file
        #: having a `.csv` file extension and contain at least two
        #: (2) columns, one (1) for `text` i.e. the user phrase
        #: (labelled fine_tuning_text_processor.data_column_title)
        #: , and one (1) for `label` i.e. the intent (labelled
        #: fine_tuning_text_processor.label_column_title)
        self.training_data: pd.DataFrame = \
            pd.read_csv(self.training_path)

        #: The `training_intents_list` attribute represents a list
        #: of unique intents found in the `self.training_data`.
        #: By definition the `training_intents_list` attribute is
        #: a Python list of all unique values found in the column
        #: with title "intent" in the `self.training_data`
        #: dataframe.
        self.training_intents_list: List[str] = \
            self.training_data.intent.unique().tolist()

        #: The `training_intents_set` attribute represents a set
        #: of unique intents found in the `self.training_data`.
        #: By definition the `training_intents_set` attribute is
        #: a Python set of values found in the column
        #: with title "intent" in the `self.training_data`
        #: dataframe.
        self.training_intents_set: Set[str] = \
            set(self.training_intents_list)

        #: The `training_intents_counts` attribute represents a
        #: set of unique intents found in the `self.training_data`
        #: . The `training_intents_counts` attribute is set by
        #: calling `self.training_data.intent.value_counts()`
        #: dataframe.
        self.training_intents_counts: object = \
            self.training_data.intent.value_counts()

        #: The `testing_data` attribute represents the fine
        #: tuning data designated as "testing data". This dataset
        #: is used to tune the hyper-parameters during the model
        #: testing iterations. This dataset should be a CSV file
        #: having a `.csv` file extension and contain at least two
        #: (2) columns, one (1) for `text` i.e. the user phrase
        #: (labelled fine_tuning_text_processor.data_column_title)
        #: , and one (1) for `label` i.e. the intent (labelled
        #: fine_tuning_text_processor.label_column_title)
        self.testing_data: pd.DataFrame = \
            pd.read_csv(self.testing_path)

        #: The `testing_intents_list` attribute represents a list
        #: of unique intents found in the `self.testing_data`.
        #: By definition the `testing_intents_list` attribute is
        #: a Python list of all unique values found in the column
        #: with title "intent" in the `self.testing_data`
        #: dataframe.
        self.testing_intents_list: List[str] = \
            self.testing_data.intent.unique().tolist()

        #: The `testing_intents_set` attribute represents a set
        #: of unique intents found in the `self.testing_data`.
        #: By definition the `testing_intents_set` attribute is
        #: a Python set of values found in the column
        #: with title "intent" in the `self.testing_data`
        #: dataframe.
        self.testing_intents_set = set(self.testing_intents_list)

        #: The `testing_intents_counts` attribute represents a
        #: set of unique intents found in the `self.testing_data`
        #: . The `testing_intents_counts` attribute is set by
        #: calling `self.testing_data.intent.value_counts()`
        #: dataframe.
        self.testing_intents_counts = \
            self.testing_data.intent.value_counts()

        #: The `evaluation_data` attribute represents the fine
        #: tuning data designated as "evaluation data". This
        #: dataset is used to tune the hyper-parameters during the
        #: model evaluation iterations. This dataset should be a
        #: CSV file having a `.csv` file extension and contain at
        #: least two (2) columns, one (1) for `text` i.e. the user
        #: phrase
        #: (labelled fine_tuning_text_processor.data_column_title)
        #: , and one (1) for `label` i.e. the intent (labelled
        #: fine_tuning_text_processor.label_column_title)
        self.evaluation_data = pd.read_csv(self.evaluation_path)

        #: The `evaluation_intents_list` attribute represents a
        #: list of unique intents found in the
        #: `self.evaluation_data`. By definition the
        #: `evaluation_intents_list` attribute is
        #: a Python list of all unique values found in the column
        #: with title "intent" in the `self.evaluation_data`
        #: dataframe.
        self.evaluation_intents_list = \
            self.evaluation_data.intent.unique().tolist()

        #: The `evaluation_intents_set` attribute represents a set
        #: of unique intents found in the `self.evaluation_data`.
        #: By definition the `evaluation_intents_set` attribute is
        #: a Python set of values found in the column
        #: with title "intent" in the `self.evaluation_data`
        #: dataframe.
        self.evaluation_intents_set = \
            set(self.evaluation_intents_list)

        #: The `evaluation_intents_counts` attribute represents a
        #: set of unique intents found in the
        #: `self.evaluation_data`. The `evaluation_intents_counts`
        #: attribute is set by calling
        #: `self.testing_data.intent.value_counts()`
        #: dataframe.
        self.evaluation_intents_counts = \
            self.evaluation_data.intent.value_counts()

        #: The `regression_data` attribute represents the fine
        #: tuning data designated as "regression data". This
        #: dataset is used to tune the hyper-parameters during the
        #: model regression iterations. This dataset should be a
        #: CSV file having a `.csv` file extension and contain at
        #: least two (2) columns, one (1) for `text` i.e. the user
        #: phrase
        #: (labelled fine_tuning_text_processor.data_column_title)
        #: , and one (1) for `label` i.e. the intent (labelled
        #: fine_tuning_text_processor.label_column_title)
        self.regression_data = pd.read_csv(self.regression_path)

        #: The `evaluation_intents_list` attribute represents a
        #: list of unique intents found in the
        #: `self.evaluation_data`. By definition the
        #: `evaluation_intents_list` attribute is
        #: a Python list of all unique values found in the column
        #: with title "intent" in the `self.evaluation_data`
        #: dataframe.
        self.regression_intents_list = \
            self.regression_data.intent.unique().tolist()

        #: The `regression_intents_set` attribute represents a set
        #: of unique intents found in the `self.regression_data`.
        #: By definition the `regression_intents_set` attribute is
        #: a Python set of values found in the column
        #: with title "intent" in the `self.regression_data`
        #: dataframe.
        self.regression_intents_set = \
            set(self.regression_intents_list)

        #: The `regression_intents_counts` attribute represents a
        #: set of unique intents found in the
        #: `self.regression_data`. The `regression_intents_counts`
        #: attribute is set by calling
        #: `self.testing_data.intent.value_counts()`
        #: dataframe.
        self.regression_intents_counts = \
            self.regression_data.intent.value_counts()

        #: The `all_intents` attribute represents a list of all
        #: unique intents present in the training, testing,
        #: evaluation, and regression datasets.
        self.all_intents = list(
            set(
                self.training_intents_list
                + self.testing_intents_list
                + self.evaluation_intents_list
                + self.regression_intents_list
            )
        )

        #: The `num_of_training_intents` attribute represents the
        #: integer number of unique intents found in the
        #: `self.training_data` dataset.
        self.num_of_training_intents: int = \
            len(self.training_intents_list)

        #: The `num_of_testing_intents` attribute represents the
        #: integer number of unique intents found in the
        #: `self.testing_data` dataset.
        self.num_of_testing_intents: int = \
            len(self.testing_intents_list)

        #: The `num_of_evaluation_intents` attribute represents
        #: the integer number of unique intents found in the
        #: `self.evaluation_data` dataset.
        self.num_of_evaluation_intents: int = \
            len(self.evaluation_intents_list)

        #: The `num_of_regression_intents` attribute represents
        #: the integer number of unique intents found in the
        #: `self.regression_data` dataset.
        self.num_of_regression_intents: int = \
            len(self.regression_intents_list)

        # It would be unusual for the set of testing intents to be
        # unequal to the set of training intents.
        if (self.num_of_training_intents
            - self.num_of_testing_intents) != 0:
            WoodgateLogger.logger.warn(
                "number of training intents "
                + f"({self.num_of_training_intents}) "
                + "not equal to number of testing intents "
                + f"({self.num_of_testing_intents})"
            )

        # It would be unusual for the set of evaluation intents
        # to be unequal to the set of training intents.
        if (self.num_of_training_intents
            - self.num_of_evaluation_intents) != 0:
            WoodgateLogger.logger.warn(
                "number of training intents "
                + f"({self.num_of_training_intents}) "
                + "not equal to number of evaluation intents "
                + f"({self.num_of_evaluation_intents})"
            )

        # It would be unusual for the set of regression intents
        # to be unequal to the set of training intents.
        if (self.num_of_training_intents
            - self.num_of_regression_intents) != 0:
            WoodgateLogger.logger.warn(
                "number of training intents "
                + f"({self.num_of_regression_intents}) "
                + "not equal to number of regression intents "
                + f"({self.num_of_evaluation_intents})"
            )

        #: The `intents_data` attribute represents a Python
        #: dictionary containing key-value pairs of the type
        #: str-List[str] or str-List[int]. These items are
        #: intent lists or intent count lists respectively.
        self.intents_data: Dict[
            str,
            Union[List[str], List[int]]
        ] = {
            "intents": self.all_intents,
            "testing_set": {
                "intent": self.testing_intents_list,
                "count": self.testing_intents_counts
            },
            "training_set": {
                "intent": self.training_intents_list,
                "count": self.training_intents_counts
            },
            "evaluation_set": {
                "intent": self.evaluation_intents_list,
                "count": self.evaluation_intents_counts
            },
            "regression_set": {
                "intent": self.regression_intents_list,
                "count": self.regression_intents_counts
            }
        }

        #: The `datasets_summary_dir` attribute represents a
        #: directory on the host's file system. This is where the
        #: summary files generated by the `create_*` methods
        #: are stored.  
        self.datasets_summary_dir: str = os.getenv(
            "DATASET_SUMMARY_DIR",
            os.path.join(
                build_configuration.build_dir,
                "dataset_summary"
            )
        )

    def create_intents_data_json(
            self
    ):
        """This method will create one (1) file containing data
        which describes the distribution of intents across fine
        tuning datasets. The file will be stored in the
        `self.datasets_summary_dir` directory in JSON format
        with `.json` file extension.

        1 - Intents Data
                * `intentsData.json`


        :return: This method returns None
        :rtype: NoneType
        """
        intents_data_json = os.path.join(
            self.datasets_summary_dir, "intentsData.json")
        with open(intents_data_json) as file:
            file.write(json.dumps(self.intents_data))

        return None

    def create_intents_bar_plots(
            self
    ):
        """The method will create four (4) bar plots describing
        the distribution of intents across the four (4) fine
        tuning datasets. The plots will be stored in the
        `self.datasets_summary_dir` directory as PNG file with
        `.png` file extensions.

        1 - Training Intents
                * `intents_bar_plot_training.png`
        2 - Testing Intents
                * `intents_bar_plot_testing.png`
        3 - Evaluation Intents
                * `intents_bar_plot_evaluation.png`
        4 - Regression Intents
                * `intents_bar_plot_regression.png`

        :return: This method returns None
        :rtype: NoneType
        """

        # Plot 1
        fig, axs = plt.subplots()
        axs.title.set_text("Training set")
        axs.barh(
            self.training_intents_list,
            self.training_intents_counts
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.datasets_summary_dir,
                "intents_bar_plot_training.png"
            )
        )
        plt.figure().clear()

        # Plot 2
        fig, axs = plt.subplots()
        axs.title.set_text("Testing set")
        axs.barh(
            self.testing_intents_list,
            self.testing_intents_counts
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.datasets_summary_dir,
                "intents_bar_plot_testing.png"
            )
        )
        plt.figure().clear()

        # Plot 3
        fig, axs = plt.subplots()
        axs.title.set_text("Evaluation set")
        axs.barh(
            self.evaluation_intents_list,
            self.evaluation_intents_counts
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.datasets_summary_dir,
                "intents_bar_plot_evaluation.png"
            )
        )
        plt.figure().clear()

        # Plot 4
        fig, axs = plt.subplots()
        axs.title.set_text("Regression set")
        axs.barh(
            self.regression_intents_list,
            self.regression_intents_counts
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.datasets_summary_dir,
                "intents_bar_plot_regression.png"
            )
        )
        plt.figure().clear()

        return None

    def create_intents_venn_diagrams(
            self
    ):
        """The method will create six (6) Venn diagrams describing
        two (2) sets and their intersections. The diagrams will be
        stored in the `self.datasets_summary_dir` directory as
        PNG files with `.png` file extensions.

        1 - Training and Evaluation
                * `intents_venn_training_evaluation.png`
        2 - Training and Testing
                * `intents_venn_training_testing.png`
        3 - Training and Regression
                * `intents_venn_training_regression.png`
        4 - Evaluation and Testing
                * `intents_venn_evaluation_testing.png`
        5 - Evaluation and Regression
                * `intents_venn_evaluation_regression.png`
        6 - Testing and Regression
                * `intents_venn_testing_regression.png`

        :return: This method returns None
        :rtype: NoneType
        """
        plt_title = "Datasets - Intents Venn Diagram - "

        # Plot 1
        plt.figure(figsize=(4, 4))
        venn2(
            [
                self.training_intents_set,
                self.evaluation_intents_set,
            ],
            ("Training", "Evaluation")
        )
        plt.title(plt_title + "Training and Evaluation")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.datasets_summary_dir,
                "intents_venn_training_evaluation.png"
            )
        )
        plt.figure().clear()

        # Plot 2
        plt.figure(figsize=(4, 4))
        venn2(
            [
                self.training_intents_set,
                self.testing_intents_set,
            ],
            ("Training", "Testing")
        )
        plt.title(plt_title + "Training and Testing")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.datasets_summary_dir,
                "intents_venn_training_testing.png"
            )
        )
        plt.figure().clear()

        # Plot 3
        plt.figure(figsize=(4, 4))
        venn2(
            [
                self.training_intents_set,
                self.regression_intents_set,
            ],
            ("Training", "Regression")
        )
        plt.title(plt_title + "Training and Regression")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.datasets_summary_dir,
                "intents_venn_training_regression.png"
            )
        )
        plt.figure().clear()

        # Plot 4
        plt.figure(figsize=(4, 4))
        venn2(
            [
                self.evaluation_intents_set,
                self.testing_intents_set,
            ],
            ("Evaluation", "Testing")
        )
        plt.title(plt_title + "Evaluation and Testing")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.datasets_summary_dir,
                "intents_venn_evaluation_testing.png"
            )
        )
        plt.figure().clear()

        # Plot 5
        plt.figure(figsize=(4, 4))
        venn2(
            [
                self.evaluation_intents_set,
                self.regression_intents_set,
            ],
            ("Evaluation", "Regression")
        )
        plt.title(plt_title + "Evaluation and Regression")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.datasets_summary_dir,
                "intents_venn_evaluation_regression.png"
            )
        )
        plt.figure().clear()

        # Plot 6
        plt.figure(figsize=(4, 4))
        venn2(
            [
                self.testing_intents_set,
                self.regression_intents_set,
            ],
            ("Testing", "Regression")
        )
        plt.title(plt_title + "Testing and Regression")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.datasets_summary_dir,
                "intents_venn_testing_regression.png"
            )
        )
        plt.figure().clear()

        return None
