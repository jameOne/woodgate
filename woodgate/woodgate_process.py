"""
woodgate_process.py - The woodgate_process.py module contains the
Woodgate class definition.
"""
import datetime

from woodgate.woodgate_logger import WoodgateLogger
from .build.file_system_configuration import \
    FileSystemConfiguration
from .fine_tuning.text_processor import TextProcessor
from .fine_tuning.dataset_retrieval_strategy import \
    DatasetRetrievalStrategy
from .model.definition import Definition
from .fine_tuning.datasets_configuration import \
    DatasetsConfiguration
from .build.build_summary import BuildSummary
from .model.evaluation import ModelEvaluation
from .model.fitter import Fitter
from .model.model_summary import ModelSummary
from .model.compiler import Compiler
from .model.storage_strategy import StorageStrategy
from .build.sanity_check import SanityCheck


class WoodgateProcess:
    """
    WoodgateProcess - The WoodgateProcess class encapsulates the
    build process. This class should have only a single static
    method `WoodgateProcess.run()` which takes no arguments.
    """

    @staticmethod
    def run():
        """

        :return:
        :rtype:
        """
        start_time = datetime.datetime.now()

        WoodgateLogger.logger.info(
            "Woodgate process started: "
            + f"{start_time.strftime('%Y-%m-%d %H:%M:%s')}"
        )

        WoodgateLogger.logger.info(
            "Retrieving training data"
        )
        DatasetRetrievalStrategy.retrieve_training_dataset(
            url=DatasetsConfiguration.training_dataset_url
        )

        WoodgateLogger.logger.info(
            "Retrieving testing data"
        )
        DatasetRetrievalStrategy.retrieve_testing_dataset(
            url=DatasetsConfiguration.testing_dataset_url
        )

        WoodgateLogger.logger.info(
            "Retrieving evaluation data"
        )
        DatasetRetrievalStrategy.retrieve_evaluation_dataset(
            url=DatasetsConfiguration.evaluation_dataset_url
        )

        WoodgateLogger.logger.info(
            "Retrieving regression data"
        )
        DatasetRetrievalStrategy.retrieve_regression_dataset(
            url=DatasetsConfiguration.regression_dataset_url
        )

        WoodgateLogger.logger.info(
            "Setting training data"
        )
        DatasetsConfiguration.set_training_data()

        WoodgateLogger.logger.info(
            "Setting testing data"
        )
        DatasetsConfiguration.set_testing_data()

        WoodgateLogger.logger.info(
            "Setting evaluation data"
        )
        DatasetsConfiguration.set_evaluation_data()

        WoodgateLogger.logger.info(
            "Setting regression data"
        )
        DatasetsConfiguration.set_regression_data()

        WoodgateLogger.logger.info(
            "Creating fine tuning dataset visuals: "
            + f"{FileSystemConfiguration.create_dataset_visuals}"
        )

        if FileSystemConfiguration.create_dataset_visuals:
            WoodgateLogger.logger.info(
                "Creating bar plots of intent "
                + "classification bins/buckets"
            )
            bar_plots_created_successfully = False
            try:
                DatasetsConfiguration.create_bar_plots()
                bar_plots_created_successfully = True
            except OSError as err:
                WoodgateLogger.logger.error(err)
            finally:
                WoodgateLogger.logger.error(
                    "An unknown error occurred while "
                    + "creating bar plots from fine tuning data"
                )
            WoodgateLogger.logger.info(
                "Bar plots of fine tuning data created:"
                + f"{bar_plots_created_successfully}"
            )

            WoodgateLogger.logger.info(
                "Creating venn diagrams of intent "
                + "classification bins/buckets"
            )
            venn_diagrams_created = False
            try:
                DatasetsConfiguration.create_venn_diagrams()
                venn_diagrams_created = True
            except OSError as err:
                WoodgateLogger.logger.error(err)
            finally:
                WoodgateLogger.logger.error(
                    "An unknown error occurred while creating "
                    + "venn diagrams from fine tuning data"
                )
            WoodgateLogger.logger.info(
                "Venn diagrams of fine tuning data created: "
                + f"{venn_diagrams_created}"
            )

        WoodgateLogger.logger.info(
            "Processing textual data for training"
        )

        data = TextProcessor(
            DatasetsConfiguration.training_data,
            DatasetsConfiguration.testing_data,
            Definition.get_tokenizer(),
            DatasetsConfiguration.all_intents()
        )

        WoodgateLogger.logger.info(
            "train x shape: "
            + f"{data.train_x.shape}"
        )
        WoodgateLogger.logger.info(
            "train x element example: "
            + f"{data.train_x[0]}"
        )
        WoodgateLogger.logger.info(
            "train y element example: "
            + f"{data.train_y[0]}"
        )
        WoodgateLogger.logger.info(
            "data max_length_sequence:"
            + f"{data.max_sequence_length}"
        )

        WoodgateLogger.logger.info("Creating BERT model")
        bert_model = Definition.create_model(
            data.max_sequence_length,
            len(DatasetsConfiguration.all_intents())
        )

        ModelSummary.print(bert_model=bert_model)

        WoodgateLogger.logger.info(
            "Compiling BERT model"
        )
        Compiler.compile(bert_model=bert_model)
        WoodgateLogger.logger.info(
            "BERT model compilation complete"
        )

        WoodgateLogger.logger.info("Generating build history")
        build_history = Fitter.fit(
            bert_model=bert_model,
            data=data
        )

        WoodgateLogger.logger.info(
            "Creating build history visuals: "
            + f"{FileSystemConfiguration.create_build_visuals}"
        )
        if FileSystemConfiguration.create_build_visuals:
            WoodgateLogger.logger.info(
                "Creating plot of accuracy vs. epochs"
            )
            accuracy_over_epochs_plot_created = False
            try:
                BuildSummary.create_accuracy_over_epochs_plot(
                    build_history=build_history
                )
                accuracy_over_epochs_plot_created = True
            except OSError as err:
                WoodgateLogger.logger.error(err)
            finally:
                WoodgateLogger.logger.error(
                    "An unknown error occurred while creating "
                    + "accuracy over epochs plot from "
                    + "build history"
                )
            WoodgateLogger.logger.info(
                "Plot of accuracy vs. epochs created:"
                + f"{accuracy_over_epochs_plot_created}"
            )

            WoodgateLogger.logger.info(
                "Creating plot of loss vs. epochs"
            )
            loss_over_epochs_plot_created = False
            try:
                BuildSummary.create_loss_over_epochs_plot(
                    build_history=build_history
                )
                loss_over_epochs_plot_created = True
            except OSError as err:
                WoodgateLogger.logger.error(err)
            finally:
                WoodgateLogger.logger.error(
                    "An unknown error occurred while creating"
                    + "loss over epochs plot from build history"
                )
            WoodgateLogger.logger.info(
                "Plot of loss vs. epochs created:"
                + f"{loss_over_epochs_plot_created}"
            )

        WoodgateLogger.logger.info(
            "Evaluating model accuracy"
        )
        ModelEvaluation.evaluate_model_accuracy(
            bert_model=bert_model,
            data=data
        )

        WoodgateLogger.logger.info(
            "Creating classification report"
        )
        ModelEvaluation.create_classification_report(
            bert_model=bert_model,
            data=data
        )

        WoodgateLogger.logger.info(
            "Creating confusion matrix"
        )
        ModelEvaluation.create_confusion_matrix(
            bert_model=bert_model,
            data=data
        )

        WoodgateLogger.logger.info(
            "Performing regression testing"
        )
        ModelEvaluation.perform_regression_testing(
            bert_model=bert_model,
            data=data
        )

        WoodgateLogger.logger.info("Saving model to disk")
        StorageStrategy.save_model(bert_model=bert_model)

        WoodgateLogger.logger.info("Checking build sanity")
        SanityCheck.check_sanity(data)
        build_duration = datetime.datetime.now() - start_time
        WoodgateLogger.logger.info(
            "Build process completed: "
            + f"{build_duration}"
        )
