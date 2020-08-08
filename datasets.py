"""
datasets.py - This module contains the Dataset class definition which encapsulates logic related to
training, testing, and validation datasets.
"""
import os
import json
import pandas as pd
from build_configuration import BuildConfiguration
from matplotlib_venn import venn2
import matplotlib.pyplot as plt


class Datasets:
    """
    Dataset - Class - The Dataset class encapsulates logic related to
    training, testing, and validation datasets.
    """

    # Read training data into dataframe
    training_data = pd.read_csv(BuildConfiguration.TRAINING_DATA)
    training_intents_list = training_data.intent.unique().tolist()
    training_intents_set = set(training_intents_list)
    training_intents_counts = training_data.intent.value_counts()

    # Read testing data into dataframe
    testing_data = pd.read_csv(BuildConfiguration.TESTING_DATA)
    testing_intents_list = testing_data.intent.unique().tolist()
    testing_intents_set = set(testing_intents_list)
    testing_intents_counts = testing_data.intent.value_counts()

    # Read validation data into dataframe
    validation_data = pd.read_csv(BuildConfiguration.VALIDATION_DATA)
    validation_intents_list = validation_data.intent.unique().tolist()
    validation_intents_set = set(validation_intents_list)
    validation_intents_counts = validation_data.intent.value_counts()

    # Read regression data into dataframe
    regression_data = pd.read_csv(BuildConfiguration.REGRESSION_DATA)
    regression_intents_list = regression_data.intent.unique().tolist()
    regression_intents_set = set(regression_intents_list)
    regression_intents_counts = regression_data.intent.value_counts()

    # TODO - There needs to be a re-definition for all intents. Basically
    #  it should be the training intent list + warnings and errors for
    #  cases people try to test, validate, or regress on intent sets that
    #  are larger than the set of training intents.
    all_intents = list(set(training_intents_list + testing_intents_list
                           + validation_intents_list + validation_intents_list))

    DATASET_SUMMARY_DIR = os.getenv("DATASET_SUMMARY_DIR",
                                    os.path.join(BuildConfiguration.OUTPUT_DIR, "dataset_summary"))

    intents_data = {
        "intents": all_intents,
        "testing_set": {
            "intent": testing_intents_list,
            "count": testing_intents_counts
        },
        "training_set": {
            "intent": training_intents_list,
            "count": training_intents_counts
        },
        "validation_set": {
            "intent": validation_intents_list,
            "count": validation_intents_counts
        }
    }

    @staticmethod
    def create_intents_data_json():
        """

        :return:
        :rtype:
        """
        intents_data_json = os.path.join(Datasets.DATASET_SUMMARY_DIR, "intentsData.json")
        with open(intents_data_json) as file:
            file.write(json.dumps(Datasets.intents_data))

    @staticmethod
    def create_intents_bar_plots():
        """

        :return:
        :rtype:
        """
        fig, axs = plt.subplots()
        axs.title.set_text("Testing set")
        axs.barh(Datasets.testing_intents_list, Datasets.testing_intents_counts)
        plt.tight_layout()
        plt.savefig(os.path.join(Datasets.DATASET_SUMMARY_DIR, "intents_bar_plot_testing.png"))
        plt.figure().clear()

        fig, axs = plt.subplots()
        axs.title.set_text("Training set")
        axs.barh(Datasets.training_intents_list, Datasets.training_intents_counts)
        plt.tight_layout()
        plt.savefig(os.path.join(Datasets.DATASET_SUMMARY_DIR, "intents_bar_plot_training.png"))
        plt.figure().clear()

        fig, axs = plt.subplots()
        axs.title.set_text("Validation set")
        axs.barh(Datasets.validation_intents_list, Datasets.validation_intents_counts)
        plt.tight_layout()
        plt.savefig(os.path.join(Datasets.DATASET_SUMMARY_DIR, "intents_bar_plot_validation.png"))
        plt.figure().clear()

        fig, axs = plt.subplots()
        axs.title.set_text("Regression set")
        axs.barh(Datasets.regression_intents_list, Datasets.regression_intents_counts)
        plt.tight_layout()
        plt.savefig(os.path.join(Datasets.DATASET_SUMMARY_DIR, "intents_bar_plot_regression.png"))
        plt.figure().clear()
        
    @staticmethod
    def create_intents_venn_diagram():
        """

        :return:
        :rtype:
        """

        plt.figure(figsize=(4, 4))
        venn2(
            [
                Datasets.training_intents_set,
                Datasets.validation_intents_set,
            ],
            ("Training", "Validation")
        )
        plt.title("Datasets - Intents Venn Diagram - Training and Validation")
        plt.tight_layout()
        plt.savefig(os.path.join(Datasets.DATASET_SUMMARY_DIR, "intents_venn_training_validation.png"))
        plt.figure().clear()

        plt.figure(figsize=(4, 4))
        venn2(
            [
                Datasets.training_intents_set,
                Datasets.testing_intents_set,
            ],
            ("Training", "Testing")
        )
        plt.title("Datasets - Intents Venn Diagram - Training and Testing")
        plt.tight_layout()
        plt.savefig(os.path.join(Datasets.DATASET_SUMMARY_DIR, "intents_venn_training_testing.png"))
        plt.figure().clear()

        plt.figure(figsize=(4, 4))
        venn2(
            [
                Datasets.training_intents_set,
                Datasets.regression_intents_set,
            ],
            ("Training", "Regression")
        )
        plt.title("Datasets - Intents Venn Diagram - Training and Regression")
        plt.tight_layout()
        plt.savefig(os.path.join(Datasets.DATASET_SUMMARY_DIR, "intents_venn_training_regression.png"))
        plt.figure().clear()

        plt.figure(figsize=(4, 4))
        venn2(
            [
                Datasets.validation_intents_set,
                Datasets.testing_intents_set,
            ],
            ("Validation", "Testing")
        )
        plt.title("Datasets - Intents Venn Diagram - Validation and Testing")
        plt.tight_layout()
        plt.savefig(os.path.join(Datasets.DATASET_SUMMARY_DIR, "intents_venn_validation_testing.png"))
        plt.figure().clear()

        plt.figure(figsize=(4, 4))
        venn2(
            [
                Datasets.validation_intents_set,
                Datasets.regression_intents_set,
            ],
            ("Validation", "Regression")
        )
        plt.title("Datasets - Intents Venn Diagram - Validation and Regression")
        plt.tight_layout()
        plt.savefig(os.path.join(Datasets.DATASET_SUMMARY_DIR, "intents_venn_validation_regression.png"))
        plt.figure().clear()

        plt.figure(figsize=(4, 4))
        venn2(
            [
                Datasets.testing_intents_set,
                Datasets.regression_intents_set,
            ],
            ("Testing", "Regression")
        )
        plt.title("Datasets - Intents Venn Diagram - Testing and Regression")
        plt.tight_layout()
        plt.savefig(os.path.join(Datasets.DATASET_SUMMARY_DIR, "intents_venn_testing_regression.png"))
        plt.figure().clear()
