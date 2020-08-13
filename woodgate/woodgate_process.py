"""
woodgate_process.py - The woodgate_process.py module contains the Woodgate class definition.
"""
import datetime

from woodgate.woodgate_logger import WoodgateLogger
from .build.build_configuration import BuildConfiguration
from .fine_tuning.fine_tuning_text_processor import FineTuningTextProcessor
from .model.model_definition import ModelDefinition
from .fine_tuning.fine_tuning_datasets import FineTuningDatasets
from .build.build_summary import BuildSummary
from .model.model_evaluation import ModelEvaluation
from .model.model_fit import ModelFit
from .model.model_summary import ModelSummary
from .model.model_compiler import ModelCompiler
from .model.model_storage_strategy import ModelStorageStrategy
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
        WoodgateLogger.logger.info("Woodgate process started:", start_time)

        WoodgateLogger.logger.info("Initializing build configuration")
        build_configuration = BuildConfiguration()
        WoodgateLogger.logger.info("Creating fine tuning dataset visuals:", build_configuration.create_dataset_visuals)



        fine_tuning_datasets = FineTuningDatasets(build_configuration=build_configuration)

        if build_configuration.create_dataset_visuals:
            WoodgateLogger.logger.info("Creating bar plots of intent classification bins/buckets")
            bar_plots_created_successfully = False
            try:
                fine_tuning_datasets.create_intents_bar_plots()
                bar_plots_created_successfully = True
            except OSError as err:
                WoodgateLogger.logger.error(err)
            finally:
                WoodgateLogger.logger.error("An unknown error occurred while creating bar plots from fine tuning data")
            WoodgateLogger.logger.info("Bar plots of fine tuning data created:", bar_plots_created_successfully)

            WoodgateLogger.logger.info("Creating venn diagrams of intent classification bins/buckets")
            venn_diagrams_created_successfully = False
            try:
                fine_tuning_datasets.create_intents_venn_diagrams()
                venn_diagrams_created_successfully = True
            except OSError as err:
                WoodgateLogger.logger.error(err)
            finally:
                WoodgateLogger.logger.error("An unknown error occurred while creating venn diagrams from fine tuning data")
            WoodgateLogger.logger.info("Venn diagrams of fine tuning data created:", venn_diagrams_created_successfully)

        WoodgateLogger.logger.info("Initializing model definition")
        model_definition = ModelDefinition(build_configuration=build_configuration)
        WoodgateLogger.logger.info("Processing textual data for training")

        data = FineTuningTextProcessor(
            fine_tuning_datasets.training_data,
            fine_tuning_datasets.testing_data,
            model_definition.tokenizer,
            fine_tuning_datasets.all_intents
        )

        WoodgateLogger.logger.info("train x shape: ", data.train_x.shape)
        WoodgateLogger.logger.info("train x element example: ", data.train_x[0])
        WoodgateLogger.logger.info("train y element example: ", data.train_y[0])
        WoodgateLogger.logger.info("data max_length_sequence", data.max_sequence_length)

        WoodgateLogger.logger.info("Creating BERT model")
        bert_model = model_definition.create_model(data.max_sequence_length, len(fine_tuning_datasets.all_intents))

        summary = ModelSummary.summarize(bert_model=bert_model)
        WoodgateLogger.logger.info("BERT model Summary:\n", summary)

        WoodgateLogger.logger.info("Compiling BERT model")
        ModelCompiler.compile(bert_model=bert_model)
        WoodgateLogger.logger.info("BERT model compilation complete")

        WoodgateLogger.logger.info("Initializing model fit")
        model_fit = ModelFit(build_configuration=build_configuration)

        WoodgateLogger.logger.info("Generating build history")
        build_history = model_fit.fit(bert_model=bert_model, data=data)

        WoodgateLogger.logger.info("Creating build history visuals:", build_configuration.create_build_visuals)
        if build_configuration.create_build_visuals:
            WoodgateLogger.logger.info("Initializing build summary")

            build_summary = BuildSummary(build_configuration=build_configuration)

            WoodgateLogger.logger.info("Creating plot of accuracy vs. epochs")
            accuracy_over_epochs_plot_created_successfully = False
            try:
                build_summary.create_accuracy_over_epochs_plot(build_history=build_history)
                accuracy_over_epochs_plot_created_successfully = True
            except OSError as err:
                WoodgateLogger.logger.error(err)
            finally:
                WoodgateLogger.logger.error(
                    "An unknown error occurred while creating accuracy over epochs plot from build history")
            WoodgateLogger.logger.info(
                "Plot of accuracy vs. epochs created:", accuracy_over_epochs_plot_created_successfully)

            WoodgateLogger.logger.info("Creating plot of loss vs. epochs")
            loss_over_epochs_plot_created_successfully = False
            try:
                build_summary.create_loss_over_epochs_plot(build_history=build_history)
                loss_over_epochs_plot_created_successfully = True
            except OSError as err:
                WoodgateLogger.logger.error(err)
            finally:
                WoodgateLogger.logger.error(
                    "An unknown error occurred while creating loss over epochs plot from build history")
            WoodgateLogger.logger.info(
                "Plot of loss vs. epochs created:", loss_over_epochs_plot_created_successfully)

        WoodgateLogger.logger.info("Initializing model evaluation")
        model_evaluation = ModelEvaluation(
            build_configuration=build_configuration,
            model_definition=model_definition,
            fine_tuning_datasets=fine_tuning_datasets
        )

        WoodgateLogger.logger.info("Evaluating model accuracy")
        model_evaluation.evaluate_model_accuracy(bert_model=bert_model, data=data)
        WoodgateLogger.logger.info("Creating classification report")
        model_evaluation.create_classification_report(bert_model=bert_model, data=data)
        WoodgateLogger.logger.info("Creating confusion matrix")
        model_evaluation.create_confusion_matrix(bert_model=bert_model, data=data)
        WoodgateLogger.logger.info("Performing regression testing")
        model_evaluation.perform_regression_testing(bert_model=bert_model, data=data)

        WoodgateLogger.logger.info("Initializing model storage strategy")
        model_storage_strategy = ModelStorageStrategy(build_configuration)
        WoodgateLogger.logger.info("Saving model to disk")
        model_storage_strategy.save_model_to_disk(bert_model=bert_model)

        WoodgateLogger.logger.info("Initializing sanity check")
        build_sanity_check = BuildSanityCheck(model_storage_strategy=model_storage_strategy)

        WoodgateLogger.logger.info("Checking build sanity")
        build_sanity_check.check_sanity(data)
        build_duration = datetime.datetime.now() - start_time
        WoodgateLogger.logger.info("Build process completed:", build_duration)
