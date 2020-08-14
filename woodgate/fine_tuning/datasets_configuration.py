"""
datasets_configuration.py - The
datasets_configuration.py module
contains the DatasetsConfiguration class definition.
"""
import os
import json
from typing import List, Set, Dict, Union
import pandas as pd
from ..build.file_system_configuration import \
    FileSystemConfiguration
from matplotlib_venn import venn2
import matplotlib.pyplot as plt
from ..woodgate_logger import WoodgateLogger


class DatasetsConfiguration:
    """
    Datasets - The Dataset class encapsulates logic
    related to training, testing, evaluation, and regression
    datasets.
    """

    #: The `training_dataset_url` attribute represents the
    #: location of the training dataset. This attribute should be
    #: a valid URL and accessible via the web. The URL will be
    #: passed to the DatasetRetrievalStrategy class whose
    #: corresponding retrieve method will be responsible for using
    #: the URL to download a copy of the dataset.
    training_dataset_url: str = os.getenv(
        "TRAINING_DATASET_URL",
        "https://drive.google.com/uc?"
        + "id=1OlcvGWReJMuyYQuOZm149vHWwPtlboR6"
    )

    #: The `testing_dataset_url` attribute represents the
    #: location of the testing dataset. This attribute should be
    #: a valid URL and accessible via the web. The URL will be
    #: passed to the DatasetRetrievalStrategy class whose
    #: corresponding retrieve method will be responsible for using
    #: the URL to download a copy of the dataset.
    testing_dataset_url: str = os.getenv(
        "TESTING_DATASET_URL",
        "https://drive.google.com/uc?"
        + "id=1ep9H6-HvhB4utJRLVcLzieWNUSG3P_uF"
    )

    #: The `evaluation_dataset_url` attribute represents the
    #: location of the evaluation dataset. This attribute should
    #: be a valid URL and accessible via the web. The URL will be
    #: passed to the DatasetRetrievalStrategy class whose
    #: corresponding retrieve method will be responsible for using
    #: the URL to download a copy of the dataset.
    evaluation_dataset_url: str = os.getenv(
        "EVALUATION_DATASET_URL",
        "https://drive.google.com/uc?"
        + "id=1Oi5cRlTybuIF2Fl5Bfsr-KkqrXrdt77w"
    )

    #: The `regression_dataset_url` attribute represents the
    #: location of the regression dataset. This attribute should
    #: be a valid URL and accessible via the web. The URL will be
    #: passed to the DatasetRetrievalStrategy class whose
    #: corresponding retrieve method will be responsible for using
    #: the URL to download a copy of the dataset.
    regression_dataset_url: str = os.getenv(
        "REGRESSION_DATASET_URL",
        "https://drive.google.com/uc?"
        + "id=1Oi5cRlTybuIF2Fl5Bfsr-KkqrXrdt77w"
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
    training_data: pd.DataFrame

    @classmethod
    def set_training_data(cls):
        """

        :return: this method returns None
        :rtype: NoneType
        """
        cls.training_data = pd.read_csv(
            FileSystemConfiguration.training_path
        )
        return None

    @classmethod
    def get_training_data(cls):
        """

        :return: this method returns the `training_data` attr.
        :rtype: pd.DataFrame
        """
        return cls.training_data

    #: The `training_intents_list` attribute represents a list
    #: of unique intents found in the `training_data`.
    #: By definition the `training_intents_list` attribute is
    #: a Python list of all unique values found in the column
    #: with title "intent" in the `training_data`
    #: dataframe.
    @classmethod
    def training_intents_list(cls):
        """

        :return: list of unique intents found in training data
        :rtype: List[str]
        """
        return cls.get_training_data().intent.unique().tolist()

    #: The `training_set` attribute represents a set
    #: of unique intents found in the `training_data`.
    #: By definition the `training_set` attribute is
    #: a Python set of values found in the column
    #: with title "intent" in the `training_data`
    #: dataframe.
    @classmethod
    def training_intents_set(cls):
        """

        :return: set of intents from training data
        :rtype: Set[str]
        """
        return set(cls.training_intents_list())

    #: The `training_intents_counts` attribute represents a
    #: set of unique intents found in the `training_data`
    #: . The `training_intents_counts` attribute is set by
    #: calling `training_data.intent.value_counts()`
    #: dataframe.
    @classmethod
    def training_intents_counts(cls):
        """

        :return:
        :rtype:
        """
        return cls.get_training_data().intent.value_counts()

    #: The `testing_data` attribute represents the fine
    #: tuning data designated as "testing data". This dataset
    #: is used to tune the hyper-parameters during the model
    #: testing iterations. This dataset should be a CSV file
    #: having a `.csv` file extension and contain at least two
    #: (2) columns, one (1) for `text` i.e. the user phrase
    #: (labelled fine_tuning_text_processor.data_column_title)
    #: , and one (1) for `label` i.e. the intent (labelled
    #: fine_tuning_text_processor.label_column_title)
    testing_data: pd.DataFrame

    @classmethod
    def set_testing_data(cls):
        """

        :return: this method returns None
        :rtype: NoneType
        """
        cls.testing_data = pd.read_csv(
            FileSystemConfiguration.testing_path
        )
        return None

    @classmethod
    def get_testing_data(cls):
        """

        :return: this method returns the `testing_data` attr.
        :rtype: pd.DataFrame
        """
        return cls.testing_data

    #: The `testing_intents_list` attribute represents a list
    #: of unique intents found in the `testing_data`.
    #: By definition the `testing_intents_list` attribute is
    #: a Python list of all unique values found in the column
    #: with title "intent" in the `testing_data`
    #: dataframe.
    @classmethod
    def testing_intents_list(cls):
        """

        :return:
        :rtype: List[str]
        """
        return cls.get_testing_data().intent.unique().tolist()

    #: The `testing_set` attribute represents a set
    #: of unique intents found in the `testing_data`.
    #: By definition the `testing_set` attribute is
    #: a Python set of values found in the column
    #: with title "intent" in the `testing_data`
    #: dataframe.
    @classmethod
    def testing_intents_set(cls):
        """

        :return: set of intents from testing data
        :rtype: Set[str]
        """
        return set(cls.testing_intents_list())

    #: The `testing_counts` attribute represents a
    #: set of unique intents found in the `testing_data`
    #: . The `testing_counts` attribute is set by
    #: calling `testing_data.intent.value_counts()`
    #: dataframe.
    @classmethod
    def testing_intents_counts(cls):
        """

        :return:
        :rtype:
        """
        return cls.get_testing_data().intent.value_counts()

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
    evaluation_data: pd.DataFrame

    @classmethod
    def set_evaluation_data(cls):
        """

        :return: this method returns None
        :rtype: NoneType
        """
        cls.evaluation_data = pd.read_csv(
            FileSystemConfiguration.evaluation_path
        )
        return None

    @classmethod
    def get_evaluation_data(cls):
        """

        :return: this method returns the `evaluation_data` attr.
        :rtype: pd.DataFrame
        """
        return cls.evaluation_data

    #: The `evaluation_list` attribute represents a
    #: list of unique intents found in the
    #: `evaluation_data`. By definition the
    #: `evaluation_list` attribute is
    #: a Python list of all unique values found in the column
    #: with title "intent" in the `evaluation_data`
    #: dataframe.
    @classmethod
    def evaluation_intents_list(cls):
        """

        :return:
        :rtype: List[str]
        """
        return cls.get_evaluation_data().intent.unique().tolist()

    #: The `evaluation_set` attribute represents a set
    #: of unique intents found in the `evaluation_data`.
    #: By definition the `evaluation_set` attribute is
    #: a Python set of values found in the column
    #: with title "intent" in the `evaluation_data`
    #: dataframe.
    @classmethod
    def evaluation_intents_set(cls):
        """

        :return: set of intents from evaluation data
        :rtype: Set[str]
        """
        return set(cls.evaluation_intents_list())

    #: The `evaluation_counts` attribute represents a
    #: set of unique intents found in the
    #: `evaluation_data`. The `evaluation_counts`
    #: attribute is set by calling
    #: `testing_data.intent.value_counts()`
    #: dataframe.
    @classmethod
    def evaluation_intents_counts(cls):
        """

        :return:
        :rtype:
        """
        return cls.get_evaluation_data().intent.value_counts()

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
    regression_data: pd.DataFrame

    @classmethod
    def set_regression_data(cls):
        """

        :return: this method returns None
        :rtype: NoneType
        """
        cls.regression_data = pd.read_csv(
            FileSystemConfiguration.regression_path
        )
        return None

    @classmethod
    def get_regression_data(cls):
        """

        :return: this method returns the `regression_data` attr.
        :rtype: pd.DataFrame
        """
        return cls.regression_data

    #: The `regression_list` attribute represents a
    #: list of unique intents found in the
    #: `regression_data`. By definition the
    #: `regression_list` attribute is
    #: a Python list of all unique values found in the column
    #: with title "intent" in the `regression_data`
    #: dataframe.
    @classmethod
    def regression_intents_list(cls):
        """

        :return:
        :rtype: List[str]
        """
        return cls.get_regression_data().intent.unique().tolist()

    #: The `regression_set` attribute represents a set
    #: of unique intents found in the `regression_data`.
    #: By definition the `regression_set` attribute is
    #: a Python set of values found in the column
    #: with title "intent" in the `regression_data`
    #: dataframe.
    @classmethod
    def regression_intents_set(cls):
        """

        :return: set of intents from regression data
        :rtype: Set[str]
        """
        return set(cls.regression_intents_list())

    #: The `regression_counts` attribute represents a
    #: set of unique intents found in the
    #: `regression_data`. The `regression_counts`
    #: attribute is set by calling
    #: `testing_data.intent.value_counts()`
    #: dataframe.
    @classmethod
    def regression_intents_counts(cls):
        """

        :return:
        :rtype:
        """
        return cls.get_regression_data().intent.value_counts()

    #: The `all_intents` attribute represents a list of all
    #: unique intents present in the training, testing,
    #: evaluation, and regression datasets.
    @classmethod
    def all_intents(cls):
        """

        :return:
        :rtype:
        """
        return list(
            set(
                cls.training_intents_list()
                + cls.testing_intents_list()
                + cls.evaluation_intents_list()
                + cls.regression_intents_list()
            )
        )

    @classmethod
    def check_intents_intersection(cls):
        """

        :return:
        :rtype:
        """
        #: The `num_of_training_intents` attribute represents the
        #: integer number of unique intents found in the
        #: `training_data` dataset.
        num_of_training_intents: int = \
            len(cls.training_intents_list())

        #: The `num_of_testing_intents` attribute represents the
        #: integer number of unique intents found in the
        #: `testing_data` dataset.
        num_of_testing_intents: int = \
            len(cls.testing_intents_list())

        #: The `num_of_evaluation_intents` attribute represents
        #: the integer number of unique intents found in the
        #: `evaluation_data` dataset.
        num_of_evaluation_intents: int = \
            len(cls.evaluation_intents_list())

        #: The `num_of_regression_intents` attribute represents
        #: the integer number of unique intents found in the
        #: `regression_data` dataset.
        num_of_regression_intents: int = \
            len(cls.regression_intents_list())

        # It would be unusual for the set of testing intents to be
        # unequal to the set of training intents.
        if num_of_training_intents \
                - num_of_testing_intents != 0:
            WoodgateLogger.logger.warn(
                "number of training intents "
                + f"({num_of_training_intents}) "
                + "not equal to number of testing intents "
                + f"({num_of_testing_intents})"
            )

        # It would be unusual for the set of evaluation intents
        # to be unequal to the set of training intents.
        if num_of_training_intents \
                - num_of_evaluation_intents != 0:
            WoodgateLogger.logger.warn(
                "number of training intents "
                + f"({num_of_training_intents}) "
                + "not equal to number of evaluation intents "
                + f"({num_of_evaluation_intents})"
            )

        # It would be unusual for the set of regression intents
        # to be unequal to the set of training intents.
        if num_of_training_intents \
                - num_of_regression_intents != 0:
            WoodgateLogger.logger.warn(
                "number of training intents "
                + f"({num_of_regression_intents}) "
                + "not equal to number of regression intents "
                + f"({num_of_evaluation_intents})"
            )

    #: The `intents_data` attribute represents a Python
    #: dictionary containing key-value pairs of the type
    #: str-List[str] or str-List[int]. These items are
    #: intent lists or intent count lists respectively.
    @classmethod
    def intents_dict(cls):
        """

        :return:
        :rtype: Dict[str, Union[List[str], List[int]]]
        """
        return {
            "intents": cls.all_intents(),
            "training_set": {
                "intent": cls.training_intents_list(),
                "count": cls.training_intents_counts()
            },
            "testing_set": {
                "intent": cls.testing_intents_list(),
                "count": cls.testing_intents_counts()
            },
            "evaluation_set": {
                "intent": cls.evaluation_intents_list(),
                "count": cls.evaluation_intents_list()
            },
            "regression_set": {
                "intent": cls.regression_intents_list(),
                "count": cls.regression_intents_counts()
            }
        }

    #: The `datasets_summary_dir` attribute represents a
    #: directory on the host's file system. This is where the
    #: summary files generated by the `create_*` methods
    #: are stored.
    datasets_summary_dir: str = os.getenv(
        "DATASET_SUMMARY_DIR",
        os.path.join(
            FileSystemConfiguration.build_dir,
            "datasets_summary"
        )
    )
    os.makedirs(datasets_summary_dir, exist_ok=True)

    @classmethod
    def create_data_json(cls):
        """This method will create one (1) file containing data
        which describes the distribution of intents across fine
        tuning datasets. The file will be stored in the
        `cls.datasets_summary_dir` directory in
        JSON format with `.json` file extension.

        1 - Intents Data
                * `intentsData.json`


        :return: This method returns None
        :rtype: NoneType
        """
        intents_data_json = os.path.join(
            cls.datasets_summary_dir,
            "intentsData.json"
        )
        with open(intents_data_json) as file:
            file.write(
                json.dumps(cls.intents_dict())
            )

        return None

    @classmethod
    def create_bar_plots(cls):
        """The method will create four (4) bar plots describing
        the distribution of intents across the four (4) fine
        tuning datasets. The plots will be stored in the
        `cls.datasets_summary_dir` directory
        as PNG files with `.png` file extensions.

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
            cls.training_intents_list(),
            cls.training_intents_counts()
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                cls.datasets_summary_dir,
                "intents_bar_plot_training.png"
            )
        )
        plt.figure().clear()

        # Plot 2
        fig, axs = plt.subplots()
        axs.title.set_text("Testing set")
        axs.barh(
            cls.testing_intents_list(),
            cls.testing_intents_counts()
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                cls.datasets_summary_dir,
                "intents_bar_plot_testing.png"
            )
        )
        plt.figure().clear()

        # Plot 3
        fig, axs = plt.subplots()
        axs.title.set_text("Evaluation set")
        axs.barh(
            cls.evaluation_intents_list(),
            cls.evaluation_intents_counts()
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                cls.datasets_summary_dir,
                "intents_bar_plot_evaluation.png"
            )
        )
        plt.figure().clear()

        # Plot 4
        fig, axs = plt.subplots()
        axs.title.set_text("Regression set")
        axs.barh(
            cls.regression_intents_list(),
            cls.regression_intents_counts()
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                cls.datasets_summary_dir,
                "intents_bar_plot_regression.png"
            )
        )
        plt.figure().clear()

        return None

    @classmethod
    def create_venn_diagrams(cls):
        """The method will create six (6) Venn diagrams describing
        two (2) sets and their intersections. The diagrams will be
        stored in the `cls.datasets_summary_dir`
        directory as PNG files with `.png` file extensions.

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
                cls.training_intents_set(),
                cls.evaluation_intents_set(),
            ],
            ("Training", "Evaluation")
        )
        plt.title(plt_title + "Training and Evaluation")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                cls.datasets_summary_dir,
                "intents_venn_training_evaluation.png"
            )
        )
        plt.figure().clear()

        # Plot 2
        plt.figure(figsize=(4, 4))
        venn2(
            [
                cls.training_intents_set(),
                cls.testing_intents_set(),
            ],
            ("Training", "Testing")
        )
        plt.title(plt_title + "Training and Testing")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                cls.datasets_summary_dir,
                "intents_venn_training_testing.png"
            )
        )
        plt.figure().clear()

        # Plot 3
        plt.figure(figsize=(4, 4))
        venn2(
            [
                cls.training_intents_set(),
                cls.regression_intents_set(),
            ],
            ("Training", "Regression")
        )
        plt.title(plt_title + "Training and Regression")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                cls.datasets_summary_dir,
                "intents_venn_training_regression.png"
            )
        )
        plt.figure().clear()

        # Plot 4
        plt.figure(figsize=(4, 4))
        venn2(
            [
                cls.evaluation_intents_set(),
                cls.testing_intents_set(),
            ],
            ("Evaluation", "Testing")
        )
        plt.title(plt_title + "Evaluation and Testing")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                cls.datasets_summary_dir,
                "intents_venn_evaluation_testing.png"
            )
        )
        plt.figure().clear()

        # Plot 5
        plt.figure(figsize=(4, 4))
        venn2(
            [
                cls.evaluation_intents_set(),
                cls.regression_intents_set(),
            ],
            ("Evaluation", "Regression")
        )
        plt.title(plt_title + "Evaluation and Regression")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                cls.datasets_summary_dir,
                "intents_venn_evaluation_regression.png"
            )
        )
        plt.figure().clear()

        # Plot 6
        plt.figure(figsize=(4, 4))
        venn2(
            [
                cls.testing_intents_set(),
                cls.regression_intents_set(),
            ],
            ("Testing", "Regression")
        )
        plt.title(plt_title + "Testing and Regression")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                cls.datasets_summary_dir,
                "intents_venn_testing_regression.png"
            )
        )
        plt.figure().clear()

        return None
