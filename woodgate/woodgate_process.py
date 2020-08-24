"""
woodgate_process.py - The woodgate_process.py module contains the
Woodgate class definition.
"""
import datetime

from woodgate.tuning.external_datasets import ExternalDatasets
from woodgate.woodgate_logger import WoodgateLogger
from woodgate.trainer.preprocessor import Preprocessor
from woodgate.tuning.dataset_retrieval_strategy import \
    DatasetRetrievalStrategy
from woodgate.trainer.evaluator import Evaluator
from woodgate.trainer.trainer import Trainer
from woodgate.compiler.compiler import Compiler
from woodgate.trainer.storage import Storage
from woodgate.transfer.bert_model_parameters import \
    BertModelParameters
from woodgate.transfer.bert_retrieval_strategy import \
    BertRetrievalStrategy
from .woodgate_settings import Model, Build, FileSystem
from .woodgate_settings import Architecture


class WoodgateProcess:
    """
    WoodgateProcess - The WoodgateProcess class encapsulates the
    build_history process. This class should have only a single
    static method `WoodgateProcess.run()` which takes no
    arguments.
    """

    @staticmethod
    def run() -> None:
        """The `run` method starts the main build_history
        process. The `WoodgateProcess.run()` method is what one
        would likely call from a `main.py` module.

        :return: None
        :rtype: NoneType
        """

        """
        Step 1 - Startup
        The first stage of the build_history process involves:
            1) Record the start time used to determine the
            duration of the build_history according to a wall
            clock.
            2) Initialize the file system such that all
            directories which are assumed to exist
        """

        start_time = datetime.datetime.now()

        WoodgateLogger.logger.info(
            "Woodgate process started: "
            + f"{start_time.strftime('%Y-%m-%d %H:%M:%s')}"
        )

        WoodgateLogger.logger.info(
            "Initializing file system configuration"
        )
        model = Model("test")
        build = Build()
        file_system = FileSystem(model, build)
        file_system.configure()

        WoodgateLogger.logger.info(
            "Initializing external datasets"
        )

        external_datasets = ExternalDatasets()

        WoodgateLogger.logger.info(
            "Retrieving training data"
        )

        DatasetRetrievalStrategy.retrieve_tuning_dataset(
            url=external_datasets.training_dataset_url,
            output=file_system.get_training_path()
        )

        WoodgateLogger.logger.info(
            "Retrieving testing data"
        )
        DatasetRetrievalStrategy.retrieve_tuning_dataset(
            url=external_datasets.testing_dataset_url,
            output=file_system.get_testing_path()

        )

        WoodgateLogger.logger.info(
            "Retrieving evaluation data"
        )
        DatasetRetrievalStrategy.retrieve_tuning_dataset(
            url=external_datasets.evaluation_dataset_url,
            output=file_system.get_evaluation_path()
        )

        WoodgateLogger.logger.info(
            "Retrieving regression data"
        )
        DatasetRetrievalStrategy.retrieve_tuning_dataset(
            url=external_datasets.regression_dataset_url,
            output=file_system.get_regression_path()
        )

        WoodgateLogger.logger.info(
            "Setting training data"
        )
        external_datasets.set_training_data(file_system)

        WoodgateLogger.logger.info(
            "Setting testing data"
        )
        external_datasets.set_testing_data(file_system)

        WoodgateLogger.logger.info(
            "Setting evaluation data"
        )
        external_datasets.set_evaluation_data(file_system)

        WoodgateLogger.logger.info(
            "Setting regression data"
        )
        external_datasets.set_regression_data(file_system)

        WoodgateLogger.logger.info(
            "Creating JSON of intent "
            + "classification bins/buckets"
        )
        external_datasets.create_intents_data_json(file_system)

        WoodgateLogger.logger.info(
            "Initializing BERT evaluator parameters"
        )

        bert_model_parameters = BertModelParameters()

        WoodgateLogger.logger.info(
            "Initializing BERT retrieval strategy"
        )

        bert_retrieval_strategy = BertRetrievalStrategy(
            bert_model_parameters=bert_model_parameters
        )

        WoodgateLogger.logger.info(
            "Retrieving BERT evaluator for transfer learning"
        )

        bert_retrieval_strategy.download_bert(file_system)

        WoodgateLogger.logger.info(
            "Processing textual data for training"
        )

        data = Preprocessor(
            external_datasets.training_data,
            external_datasets.testing_data,
            file_system.get_bert_vocab_path(),
            external_datasets.all_intents()
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

        architecture = Architecture(
            clf_out_dropout_rate=0.5,
            clf_out_activation="tanh",
            logits_dropout_rate=0.5,
            logits_activation="softmax"
        )

        WoodgateLogger.logger.info("Creating BERT evaluator")
        bert_model = Trainer.model_factory(
            model.model_name,
            external_datasets,
            data,
            architecture,
            file_system,
        )

        WoodgateLogger.logger.info(
            "Printing summary of BERT evaluator"
        )
        bert_model.summary()

        WoodgateLogger.logger.info(
            "Compiling BERT evaluator"
        )

        optimizer = Compiler.optimizer_factory(
            name="Adam",
            learning_rate=1e-5
        )

        loss = Compiler.loss_factory(
            "Binary_Crossentropy",
            *["true", "0.5"]
        )

        metrics = Compiler.metrics_factory("binary_crossentropy")

        Compiler.compile(
            model=bert_model,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

        WoodgateLogger.logger.info(
            "BERT evaluator compilation complete"
        )

        WoodgateLogger.logger.info(
            "Initializing evaluator fitter"
        )

        trainer = Trainer(
            validation_split=0.1,
            batch_size=16,
            epochs=1,
        )

        WoodgateLogger.logger.info(
            "Generating build_history history"
        )
        build_history = trainer.fit(
            bert_model=bert_model,
            data=data
        )

        WoodgateLogger.logger.info(
            "Creating build history JSON"
        )
        trainer.create_build_history_json(
            build_history=build_history,
            file_system=file_system
        )

        WoodgateLogger.logger.info(
            "Evaluating evaluator accuracy"
        )
        Evaluator.evaluate_model_accuracy(
            model=bert_model,
            data=data
        )

        WoodgateLogger.logger.info(
            "Performing regression testing"
        )
        Evaluator.perform_regression_testing(
            model=bert_model,
            data=data,
            file_system=file_system
        )

        WoodgateLogger.logger.info(
            "Creating regression test results JSON"
        )
        Evaluator.create_regression_test_results_json(
            file_system
        )

        WoodgateLogger.logger.info("Saving evaluator to disk")
        Storage.save_model(
            bert_model=bert_model,
            file_system=file_system
        )

        build_duration = datetime.datetime.now() - start_time
        WoodgateLogger.logger.info(
            "Build process completed: "
            + f"{build_duration}"
        )

        return None


if __name__ == "__main__":
    WoodgateProcess.run()
