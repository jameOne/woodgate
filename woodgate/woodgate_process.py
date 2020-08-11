"""
woodgate_process.py - The woodgate_process.py module contains the Woodgate class definition.
"""
import datetime

from .build.build_logger import BuildLogger
from .build.build_configuration import BuildConfiguration
from .fine_tuning.fine_tuning_text_processor import FineTuningTextProcessor
from .model.model_definition import ModelDefinition
from .fine_tuning.fine_tuning_datasets import FineTuningDatasets
from .build.build_summary import BuildSummary
from .model.model_evaluation import ModelEvaluation
from .model.model_fit import ModelFit
from .model.model_summary import ModelSummary
from .model.model_compiler import ModelCompiler
from .model.model_storage import ModelStorage
from .build.build_sanity_check import BuildSanityCheck


class WoodgateProcess:
    """
    WoodgateProcess - The WoodgateProcess class encapsulates the build process. This class should have only
    a single static method `WoodgateProcess.run()` which takes no arguments.
    """

    build_configuration = BuildConfiguration()

    @staticmethod
    def run():
        """

        :return:
        :rtype:
        """
        start_time = datetime.datetime.now()
        BuildLogger.LOGGER.info("woodgate process started:", start_time)
        build_configuration = BuildConfiguration()
        BuildLogger.LOGGER.info("Creating fine tuning dataset visuals:", build_configuration.create_dataset_visuals)
        if build_configuration.create_dataset_visuals:
            BuildLogger.LOGGER.info("Creating bar plots of intent classification bins/buckets")
            bar_plots_created_successfully = False
            try:
                FineTuningDatasets.create_intents_bar_plots()
                bar_plots_created_successfully = True
            except OSError as err:
                BuildLogger.LOGGER.error(err)
            finally:
                BuildLogger.LOGGER.error("An unknown error occurred while creating bar plots from fine tuning data")
            BuildLogger.LOGGER.info("Bar plots of fine tuning data created:", bar_plots_created_successfully)

            BuildLogger.LOGGER.info("Creating venn diagrams of intent classification bins/buckets")
            venn_diagrams_created_successfully = False
            try:
                FineTuningDatasets.create_intents_venn_diagrams()
                venn_diagrams_created_successfully = True
            except OSError as err:
                BuildLogger.LOGGER.error(err)
            finally:
                BuildLogger.LOGGER.error("An unknown error occurred while creating venn diagrams from fine tuning data")
            BuildLogger.LOGGER.info("Venn diagrams of fine tuning data created:", venn_diagrams_created_successfully)

        BuildLogger.LOGGER.info("Processing textual data for training")
        data = FineTuningTextProcessor(
            FineTuningDatasets.training_data,
            FineTuningDatasets.testing_data,
            ModelDefinition.tokenizer,
            FineTuningDatasets.all_intents
        )

        BuildLogger.LOGGER.info("train x shape: ", data.train_x.shape)
        BuildLogger.LOGGER.info("train x element example: ", data.train_x[0])
        BuildLogger.LOGGER.info("train y element example: ", data.train_y[0])
        BuildLogger.LOGGER.info("data max_length_sequence", data.max_sequence_length)

        BuildLogger.LOGGER.info("Creating BERT model")
        bert_model = ModelDefinition.create_model(data.max_sequence_length, len(FineTuningDatasets.all_intents))

        summary = ModelSummary.summarize(bert_model=bert_model)
        BuildLogger.LOGGER.info("BERT model Summary:\n", summary)

        BuildLogger.LOGGER.info("Compiling BERT model")
        ModelCompiler.compile(bert_model=bert_model)
        BuildLogger.LOGGER.info("BERT model compilation complete")

        BuildLogger.LOGGER.info("Generating build history")
        build_history = ModelFit.fit(bert_model=bert_model, data=data)

        BuildLogger.LOGGER.info("Creating build history visuals:", build_configuration.create_build_visuals)
        if build_configuration.create_build_visuals:
            BuildLogger.LOGGER.info("Creating plot of accuracy vs. epochs")
            accuracy_over_epochs_plot_created_successfully = False
            try:
                BuildSummary.create_accuracy_over_epochs_plot(build_history=build_history)
                accuracy_over_epochs_plot_created_successfully = True
            except OSError as err:
                BuildLogger.LOGGER.error(err)
            finally:
                BuildLogger.LOGGER.error(
                    "An unknown error occurred while creating accuracy over epochs plot from build history")
            BuildLogger.LOGGER.info(
                "Plot of accuracy vs. epochs created:", accuracy_over_epochs_plot_created_successfully)

            BuildLogger.LOGGER.info("Creating plot of loss vs. epochs")
            loss_over_epochs_plot_created_successfully = False
            try:
                BuildSummary.create_loss_over_epochs_plot(build_history=build_history)
                loss_over_epochs_plot_created_successfully = True
            except OSError as err:
                BuildLogger.LOGGER.error(err)
            finally:
                BuildLogger.LOGGER.error(
                    "An unknown error occurred while creating loss over epochs plot from build history")
            BuildLogger.LOGGER.info(
                "Plot of loss vs. epochs created:", loss_over_epochs_plot_created_successfully)

        BuildLogger.LOGGER.info("Starting model evaluation")
        BuildLogger.LOGGER.info("Evaluating model accuracy")
        ModelEvaluation.evaluate_model_accuracy(bert_model=bert_model, data=data)
        BuildLogger.LOGGER.info("Creating classification report")
        ModelEvaluation.create_classification_report(bert_model=bert_model, data=data)
        BuildLogger.LOGGER.info("Creating confusion matrix")
        ModelEvaluation.create_confusion_matrix(bert_model=bert_model, data=data)
        BuildLogger.LOGGER.info("Performing regression testing")
        ModelEvaluation.perform_regression_testing(bert_model=bert_model, data=data)
        BuildLogger.LOGGER.info("Storing model on disk")
        ModelStorage.save_model_to_disk(bert_model=bert_model)

        BuildLogger.LOGGER.info("Sanity checking build process")
        BuildSanityCheck.check_sanity(data)
        build_duration = datetime.datetime.now() - start_time
        BuildLogger.LOGGER.info("Build process completed:", build_duration)
