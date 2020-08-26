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
from .woodgate_settings import FileSystem, Model
from .woodgate_settings import Architecture


class WoodgateProcess:
    """
    WoodgateProcess - The WoodgateProcess class encapsulates the
    build_history process. This class should have only a single
    static method `WoodgateProcess.run()` which takes no
    arguments.
    """

    @staticmethod
    def run(model: Model, file_system: FileSystem) -> None:
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
        woodgate_logger = WoodgateLogger(
            file_system=file_system
        )
        logger = woodgate_logger.logger

        start_time = datetime.datetime.now()

        logger.info(
            "Woodgate process started: "
            + f"{start_time.strftime('%Y-%m-%d %H:%M:%s')}"
        )

        logger.info(
            "Initializing file system configuration"
        )

        logger.info(
            "Initializing external datasets"
        )

        external_datasets = ExternalDatasets()

        logger.info(
            "Retrieving training data"
        )

        DatasetRetrievalStrategy.retrieve_dataset(
            file_system=file_system,
            file_id=external_datasets.training_dataset_id,
            output=file_system.get_training_path()
        )

        logger.info(
            "Retrieving testing data"
        )
        DatasetRetrievalStrategy.retrieve_dataset(
            file_system=file_system,
            file_id=external_datasets.testing_dataset_id,
            output=file_system.get_testing_path()

        )

        logger.info(
            "Retrieving evaluation data"
        )
        DatasetRetrievalStrategy.retrieve_dataset(
            file_system=file_system,
            file_id=external_datasets.evaluation_dataset_id,
            output=file_system.get_evaluation_path()
        )

        logger.info(
            "Retrieving regression data"
        )
        DatasetRetrievalStrategy.retrieve_dataset(
            file_system=file_system,
            file_id=external_datasets.regression_dataset_id,
            output=file_system.get_regression_path()
        )

        logger.info(
            "Setting training data"
        )
        external_datasets.set_training_data(file_system)

        logger.info(
            "Setting testing data"
        )
        external_datasets.set_testing_data(file_system)

        logger.info(
            "Setting evaluation data"
        )
        external_datasets.set_evaluation_data(file_system)

        logger.info(
            "Setting regression data"
        )
        external_datasets.set_regression_data(file_system)

        logger.info(
            "Creating JSON of intent "
            + "classification bins/buckets"
        )
        external_datasets.create_intents_data_json(file_system)

        logger.info(
            "Initializing BERT evaluator parameters"
        )

        bert_model_parameters = BertModelParameters()

        logger.info(
            "Initializing BERT retrieval strategy"
        )

        bert_retrieval_strategy = BertRetrievalStrategy(
            bert_model_parameters=bert_model_parameters
        )

        logger.info(
            "Retrieving BERT evaluator for transfer learning"
        )

        bert_retrieval_strategy.download_bert(file_system)

        logger.info(
            "Processing textual data for training"
        )

        data = Preprocessor(
            external_datasets.training_data,
            external_datasets.testing_data,
            file_system.get_bert_vocab_path(),
            external_datasets.all_intents()
        )

        logger.info(
            "train x shape: "
            + f"{data.train_x.shape}"
        )
        logger.info(
            "train x element example: "
            + f"{data.train_x[0]}"
        )
        logger.info(
            "train y element example: "
            + f"{data.train_y[0]}"
        )
        logger.info(
            "data max_length_sequence: "
            + f"{data.max_sequence_length}"
        )

        architecture = Architecture(
            clf_out_dropout_rate=0.5,
            clf_out_activation="tanh",
            logits_dropout_rate=0.5,
            logits_activation="softmax"
        )

        logger.info("Creating BERT evaluator")
        bert_model = Trainer.model_factory(
            model.model_name,
            external_datasets,
            data,
            architecture,
            file_system,
        )

        logger.info(
            "Printing summary of BERT evaluator"
        )
        bert_model.summary()

        logger.info(
            "Compiling BERT evaluator"
        )

        optimizer = Compiler.optimizer_factory(
            name="Adam",
            learning_rate=1e-5
        )

        loss = Compiler.loss_factory(
            "Sparse_Categorical_Crossentropy",
            *["true", "0.5"]
        )

        metrics = Compiler.metrics_factory(
            "sparse_categorical_accuracy")

        Compiler.compile(
            model=bert_model,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

        logger.info(
            "BERT evaluator compilation complete"
        )

        logger.info(
            "Initializing evaluator fitter"
        )

        trainer = Trainer(
            validation_split=0.1,
            batch_size=16,
            epochs=1,
        )

        logger.info(
            "Generating build_history history"
        )
        build_history = trainer.fit(
            bert_model=bert_model,
            data=data
        )

        logger.info(
            "Creating build history JSON"
        )
        trainer.create_build_history_json(
            build_history=build_history,
            file_system=file_system
        )

        logger.info(
            "Evaluating evaluator accuracy"
        )
        Evaluator.evaluate_model_accuracy(
            model=bert_model,
            data=data
        )

        logger.info(
            "Performing regression testing"
        )
        Evaluator.perform_regression_testing(
            model=bert_model,
            data=data,
            file_system=file_system
        )

        logger.info(
            "Creating regression test results JSON"
        )
        Evaluator.create_regression_test_results_json(
            file_system
        )

        logger.info("Saving evaluator to disk")
        Storage.save_model(
            bert_model=bert_model,
            file_system=file_system
        )

        build_duration = datetime.datetime.now() - start_time
        logger.info(
            "Build process completed: "
            + f"{build_duration}"
        )

        return None
