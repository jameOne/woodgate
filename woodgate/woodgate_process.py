"""
woodgate_process.py - The woodgate_process.py module contains the
Woodgate class definition.
"""
import datetime

from woodgate.woodgate_logger import WoodgateLogger
from .build.file_system_configuration import \
    FileSystemConfiguration
from .tuning.text_processor import TextProcessor
from .tuning.dataset_retrieval_strategy import \
    DatasetRetrievalStrategy
from .model.definition import Definition
from .tuning.external_datasets import \
    ExternalDatasets
from .build.build_summary import BuildSummary
from .model.evaluation import ModelEvaluation
from .model.fitter import Fitter
from .model.compiler import Compiler
from .model.storage_strategy import StorageStrategy
from .transfer.bert_model_parameters import BertModelParameters
from .transfer.bert_retrieval_strategy import \
    BertRetrievalStrategy
from .woodgate_settings import WoodgateSettings


class WoodgateProcess:
    """
    WoodgateProcess - The WoodgateProcess class encapsulates the
    build process. This class should have only a single static
    method `WoodgateProcess.run()` which takes no arguments.
    """

    @staticmethod
    def run() -> None:
        """The `run` method starts the main build process. The
        `WoodgateProcess.run()` method is what one would likely
        call from a `main.py` module.

        :return: None
        :rtype: NoneType
        """
        start_time = datetime.datetime.now()

        WoodgateLogger.logger.info(
            "Woodgate process started: "
            + f"{start_time.strftime('%Y-%m-%d %H:%M:%s')}"
        )

        WoodgateLogger.logger.info(
            "Initializing file system configuration"
        )
        FileSystemConfiguration(
            woodgate_settings=WoodgateSettings
        )

        WoodgateLogger.logger.info(
            "Retrieving training data"
        )
        DatasetRetrievalStrategy.retrieve_tuning_dataset(
            url=ExternalDatasets.training_dataset_url,
            output=WoodgateSettings.get_training_path()
        )

        WoodgateLogger.logger.info(
            "Retrieving testing data"
        )
        DatasetRetrievalStrategy.retrieve_tuning_dataset(
            url=ExternalDatasets.testing_dataset_url,
            output=WoodgateSettings.get_testing_path()

        )

        WoodgateLogger.logger.info(
            "Retrieving evaluation data"
        )
        DatasetRetrievalStrategy.retrieve_tuning_dataset(
            url=ExternalDatasets.evaluation_dataset_url,
            output=WoodgateSettings.get_evaluation_path()
        )

        WoodgateLogger.logger.info(
            "Retrieving regression data"
        )
        DatasetRetrievalStrategy.retrieve_tuning_dataset(
            url=ExternalDatasets.regression_dataset_url,
            output=WoodgateSettings.get_regression_path()
        )

        WoodgateLogger.logger.info(
            "Setting training data"
        )
        ExternalDatasets.set_training_data()

        WoodgateLogger.logger.info(
            "Setting testing data"
        )
        ExternalDatasets.set_testing_data()

        WoodgateLogger.logger.info(
            "Setting evaluation data"
        )
        ExternalDatasets.set_evaluation_data()

        WoodgateLogger.logger.info(
            "Setting regression data"
        )
        ExternalDatasets.set_regression_data()

        WoodgateLogger.logger.info(
            "Creating JSON of intent "
            + "classification bins/buckets"
        )
        ExternalDatasets.create_intents_data_json()

        if WoodgateSettings.create_dataset_visuals:
            WoodgateLogger.logger.info(
                "Creating bar plots of intent "
                + "classification bins/buckets"
            )
            ExternalDatasets.create_intents_bar_plots()

            WoodgateLogger.logger.info(
                "Creating venn diagrams of intent "
                + "classification bins/buckets"
            )
            ExternalDatasets.create_intents_venn_diagrams()

        WoodgateLogger.logger.info(
            "Initializing BERT model parameters"
        )

        bert_model_parameters = BertModelParameters()

        WoodgateLogger.logger.info(
            "Initializing BERT retrieval strategy"
        )

        bert_retrieval_strategy = BertRetrievalStrategy(
            bert_model_parameters=bert_model_parameters
        )

        WoodgateLogger.logger.info(
            "Retrieving BERT model for transfer learning"
        )

        bert_retrieval_strategy.download_bert()

        WoodgateLogger.logger.info(
            "Processing textual data for training"
        )

        data = TextProcessor(
            ExternalDatasets.training_data,
            ExternalDatasets.testing_data,
            Definition.get_tokenizer(),
            ExternalDatasets.all_intents()
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
            "data max_length_sequence: "
            + f"{data.max_sequence_length}"
        )

        WoodgateLogger.logger.info("Creating BERT model")
        bert_model = Definition.create_model(
            data.max_sequence_length,
            len(ExternalDatasets.all_intents())
        )

        WoodgateLogger.logger.info(
            "Printing summary of BERT model"
        )
        bert_model.summary()

        WoodgateLogger.logger.info(
            "Compiling BERT model"
        )
        Compiler.compile(bert_model=bert_model)
        WoodgateLogger.logger.info(
            "BERT model compilation complete"
        )

        WoodgateLogger.logger.info(
            "Initializing model fitter"
        )
        model_fitter = Fitter(
            woodgate_settings=WoodgateSettings
        )

        WoodgateLogger.logger.info(
            "Generating build history"
        )
        build_history = model_fitter.fit(
            bert_model=bert_model,
            data=data
        )

        WoodgateLogger.logger.info(
            "Creating accuracy vs. epochs JSON"
        )
        BuildSummary.create_accuracy_over_epochs_json(
            build_history=build_history
        )

        WoodgateLogger.logger.info(
            "Creating loss vs. epochs JSON"
        )
        BuildSummary.create_loss_over_epochs_json(
            build_history=build_history
        )

        if WoodgateSettings.create_build_visuals:
            WoodgateLogger.logger.info(
                "Creating plot of accuracy vs. epochs"
            )
            BuildSummary.create_accuracy_over_epochs_plot(
                build_history=build_history
            )

            WoodgateLogger.logger.info(
                "Creating plot of loss vs. epochs"
            )
            BuildSummary.create_loss_over_epochs_plot(
                build_history=build_history
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

        WoodgateLogger.logger.info(
            "Creating regression test results CSV"
        )
        ModelEvaluation.create_regression_test_results_csv()

        WoodgateLogger.logger.info(
            "Creating regression test results JSON"
        )
        ModelEvaluation.create_regression_test_results_json()

        if WoodgateSettings.create_evaluation_visuals:
            WoodgateLogger.logger.info(
                "Creating pie chart of regression test results"
            )
            ModelEvaluation\
                .create_regression_test_results_pie_chart()

        WoodgateLogger.logger.info("Saving model to disk")
        StorageStrategy.save_model(bert_model=bert_model)

        # WoodgateLogger.logger.info("Checking build sanity")
        # SanityCheck.check_sanity(data)
        build_duration = datetime.datetime.now() - start_time
        WoodgateLogger.logger.info(
            "Build process completed: "
            + f"{build_duration}"
        )

        return None
