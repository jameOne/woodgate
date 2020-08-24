"""
trainer_test.py - The trainer_test.py module contains all
unit tests related to the woodgate.evaluator.trainer module.
"""
import os
import glob
import unittest
import shutil
import pandas as pd
from tensorflow import keras
from woodgate.woodgate_settings import FileSystem, Model, Build
from woodgate.transfer.bert_model_parameters import \
    BertModelParameters
from woodgate.transfer.bert_retrieval_strategy import \
    BertRetrievalStrategy
from woodgate.tuning.external_datasets import ExternalDatasets
from woodgate.trainer.preprocessor import Preprocessor
from woodgate.compiler.compiler import Compiler
from ..woodgate_settings import Architecture
from ..trainer.trainer import Trainer
from .evaluator import Evaluator
from .storage import Storage


class TestTrainer(unittest.TestCase):

    def setUp(self) -> None:
        """

        :return:
        :rtype:
        """
        model = Model("test")
        build = Build()
        file_system = FileSystem(model, build)
        file_system.configure()
        self.file_system = file_system

        bert_model_parameters = BertModelParameters(
            bert_h_param=128,
            bert_l_param=2
        )

        bert_retrieval_strategy = BertRetrievalStrategy(
            bert_model_parameters=bert_model_parameters
        )

        bert_retrieval_strategy.download_bert(
            file_system=file_system
        )

        self.assertTrue(
            glob.glob(
                f"{file_system.get_bert_model_path()}*"
            )
        )

        self.file_system = file_system

        test = pd.DataFrame({
            "text": [
                "test intent john and test intent",
                "test intent jacob and test intent",
                "test intent jingle and  dfg test intent",
                "test intent sdf and test intent",
                "test intent and sdf test intent",
                "test intent and brp test intent",
                "test intent doug and test intent",
                "test intent and crow test intent",
                "test intent and all test intent",
                "test intent and test intent"
            ],
            "intent": [
                "TestIntent0",
                "TestIntent0",
                "TestIntent0",
                "TestIntent0",
                "TestIntent0",
                "TestIntent0",
                "TestIntent0",
                "TestIntent0",
                "TestIntent0",
                "TestIntent0"
            ]
        })

        self.intents = ["TestIntent0"]

        with open(
                self.file_system.get_testing_path(), "w+"
        ) as file:
            file.write(test.to_csv(index_label=False))
        ExternalDatasets.set_testing_data(self.file_system)

        with open(
                self.file_system.get_training_path(), "w+"
        ) as file:
            file.write(test.to_csv(index_label=False))
        ExternalDatasets.set_training_data(self.file_system)

        with open(
                self.file_system.get_evaluation_path(), "w+"
        ) as file:
            file.write(test.to_csv(index_label=False))
        ExternalDatasets.set_evaluation_data(self.file_system)

        with open(
                self.file_system.get_regression_path(), "w+"
        ) as file:
            file.write(test.to_csv(index_label=False))
        ExternalDatasets.set_regression_data(self.file_system)

        self.data = Preprocessor(
            test,
            test,
            self.file_system.get_bert_vocab_path(),
            self.intents
        )

        architecture = Architecture(
            clf_out_dropout_rate=0.5,
            clf_out_activation="tanh",
            logits_dropout_rate=0.5,
            logits_activation="softmax"
        )

        self.test_model = Trainer.model_factory(
            name=model.model_name,
            external_datasets=ExternalDatasets(),
            preprocessor=self.data,
            architecture=architecture,
            file_system=file_system
        )

        optimizer = Compiler.optimizer_factory(
            name="Adam",
            learning_rate=1e-5
        )

        loss = Compiler.loss_factory(
            "Binary_Crossentropy",
            *["true", "0.5"]
        )

        names = [
            "binary_crossentropy",
        ]

        metrics = Compiler.metrics_factory(*names)

        Compiler.compile(
            model=self.test_model,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

    @classmethod
    def tearDownClass(cls) -> None:
        """

        :return:
        :rtype:
        """
        woodgate_base_dir = os.path.join(
            os.path.expanduser("~"),
            "woodgate"
        )
        shutil.rmtree(woodgate_base_dir)

    def test_preprocessor(self) -> None:
        """

        :return:
        """
        self.assertEqual(self.data.max_sequence_length, 11)
        self.assertListEqual(self.data.intents, self.intents)
        self.assertListEqual(
            list(self.data.train_x[0]),
            [
                101,
                3231,
                7848,
                2198,
                1998,
                3231,
                7848,
                102,
                0,
                0,
                0
            ]
        )
        self.assertListEqual(
            list(self.data.train_y),
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        )

    def test_fit_w_tensorboard_callback(self) -> None:
        """

        :return:
        :rtype:
        """
        trainer = Trainer(
            0.1,
            16,
            1,
            True,
            self.file_system.log_dir
        )

        build_history = trainer.fit(
            self.test_model,
            self.data
        )

        self.assertIsNotNone(build_history)

        Trainer.create_build_history_json(
            build_history,
            self.file_system
        )

        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    self.file_system.build_dir,
                    "buildHistory.json"
                )
            )
        )

    def test_fit_wo_tensorboard_callback(self) -> None:
        """

        :return:
        :rtype:
        """
        trainer = Trainer(
            0.1,
            16,
            1,
            True,
            self.file_system.log_dir
        )

        build_history = trainer.fit(
            self.test_model,
            self.data
        )

        self.assertIsNotNone(build_history)

        Trainer.create_build_history_json(
            build_history,
            self.file_system
        )

        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    self.file_system.build_dir,
                    "buildHistory.json"
                )
            )
        )

    def test_evaluator_creates_regression_json(self) -> None:
        """

        :return:
        :rtype:
        """
        trainer = Trainer(
            0.1,
            16,
            1,
            True,
            self.file_system.log_dir
        )

        build_history = trainer.fit(
            self.test_model,
            self.data
        )

        self.assertIsNotNone(build_history)

        Trainer.create_build_history_json(
            build_history,
            self.file_system
        )

        Evaluator.create_regression_test_results_json(
            self.file_system
        )

        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    self.file_system.evaluation_summary_dir,
                    "regressionTestResults.json"
                )
            )
        )

    def test_save_and_load_model(self) -> None:
        """

        :return:
        :rtype:
        """
        trainer = Trainer(
            0.1,
            16,
            1,
            True,
            self.file_system.log_dir
        )

        _ = trainer.fit(
            self.test_model,
            self.data
        )

        Storage.save_model(
            self.test_model, self.file_system)

        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    self.file_system.build_dir,
                    "saved_model.pb"
                )
            )
        )

        loaded_model = Storage.load_model(
            self.file_system)

        self.assertTrue(
            isinstance(loaded_model, keras.Model)
        )


if __name__ == '__main__':
    unittest.main()
