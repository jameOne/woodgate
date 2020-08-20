"""
external_datasets_test.py - Module - The external_datasets_test.py
module contains all unit tests related to the external_datasets.py
module.
"""
import os
import json
import unittest
import pandas as pd
import shutil
from .external_datasets import ExternalDatasets
from ..woodgate_settings import \
    WoodgateSettings


class TestExternalDatasetsDefaults(unittest.TestCase):
    """
    TestExternalDatasetsDefaults - This class encapsulates all
    logic related to unit testing the ExternalDatasets class using
    default file system configuration.
    """

    def setUp(self) -> None:
        """

        :return:
        :rtype:
        """
        ExternalDatasets()
        os.makedirs(
            os.path.dirname(
                WoodgateSettings.get_training_path()
            ),
            exist_ok=True
        )
        with open(
                WoodgateSettings.get_training_path(), "w+"
        ) as file:
            file.writelines([
                "text,intent\n",
                "testTrain,TestTrainIntent\n"
            ])

        os.makedirs(
            os.path.dirname(
                WoodgateSettings.get_testing_path()
            ),
            exist_ok=True
        )
        with open(
                WoodgateSettings.get_testing_path(), "w+"
        ) as file:
            file.writelines([
                "text,intent\n",
                "testTest,TestTestIntent\n"
            ])

        os.makedirs(
            os.path.dirname(
                WoodgateSettings.get_evaluation_path()
            ),
            exist_ok=True
        )
        with open(
                WoodgateSettings.get_evaluation_path(), "w+"
        ) as file:
            file.writelines([
                "text,intent\n",
                "testEvaluate,TestEvaluateIntent\n"
            ])

        os.makedirs(
            os.path.dirname(
                WoodgateSettings.get_regression_path()
            ),
            exist_ok=True
        )
        with open(
                WoodgateSettings.get_regression_path(), "w+"
        ) as file:
            file.writelines([
                "text,intent\n",
                "testRegress,TestRegressIntent\n"
            ])

        os.makedirs(
            WoodgateSettings.datasets_summary_dir,
            exist_ok=True
        )

    def tearDown(self) -> None:
        """

        :return:
        :rtype:
        """
        os.remove(WoodgateSettings.get_training_path())
        os.remove(WoodgateSettings.get_testing_path())
        os.remove(WoodgateSettings.get_evaluation_path())
        os.remove(WoodgateSettings.get_regression_path())
        shutil.rmtree(
            WoodgateSettings.datasets_summary_dir)

    def test_default_values(self) -> None:
        """

        :return:
        :rtype:
        """
        exp_training_dataset_url = (
                "https://drive.google.com/uc?"
                + "id=1OlcvGWReJMuyYQuOZm149vHWwPtlboR6"
        )

        # make sure training dataset url has a value
        self.assertEqual(
            ExternalDatasets.training_dataset_url,
            exp_training_dataset_url
        )

        exp_testing_dataset_url = (
                "https://drive.google.com/uc?"
                + "id=1ep9H6-HvhB4utJRLVcLzieWNUSG3P_uF"
        )

        # make sure testing dataset url has a value
        self.assertEqual(
            ExternalDatasets.testing_dataset_url,
            exp_testing_dataset_url
        )

        exp_evaluation_dataset_url = (
                "https://drive.google.com/uc?"
                + "id=1Oi5cRlTybuIF2Fl5Bfsr-KkqrXrdt77w"
        )

        # make sure evaluation dataset url has a value
        self.assertEqual(
            ExternalDatasets.evaluation_dataset_url,
            exp_evaluation_dataset_url
        )

        exp_regression_dataset_url = (
                "https://drive.google.com/uc?"
                + "id=1Oi5cRlTybuIF2Fl5Bfsr-KkqrXrdt77w"
        )

        # make sure regression dataset url has a value
        self.assertEqual(
            ExternalDatasets.regression_dataset_url,
            exp_regression_dataset_url
        )

    def test_data_setters(self) -> None:
        """

        :return:
        :rtype:
        """
        exp_training_data = pd.DataFrame({
            "text": [
                "testTrain"
            ],
            "intent": [
                "TestTrainIntent"
            ]
        })
        ExternalDatasets.set_training_data()
        training_data = ExternalDatasets.get_training_data()
        self.assertTrue(
            training_data.equals(
                exp_training_data
            )
        )

        exp_testing_data = pd.DataFrame({
            "text": [
                "testTest"
            ],
            "intent": [
                "TestTestIntent"
            ]
        })
        ExternalDatasets.set_testing_data()
        testing_data = ExternalDatasets.get_testing_data()
        self.assertTrue(
            testing_data.equals(
                exp_testing_data
            )
        )

        exp_evaluation_data = pd.DataFrame({
            "text": [
                "testEvaluate"
            ],
            "intent": [
                "TestEvaluateIntent"
            ]
        })
        ExternalDatasets.set_evaluation_data()
        evaluation_data = ExternalDatasets.get_evaluation_data()
        self.assertTrue(
            evaluation_data.equals(
                exp_evaluation_data
            )
        )

        exp_regression_data = pd.DataFrame({
            "text": [
                "testRegress"
            ],
            "intent": [
                "TestRegressIntent"
            ]
        })
        ExternalDatasets.set_regression_data()
        regression_data = ExternalDatasets.get_regression_data()
        self.assertTrue(
            regression_data.equals(
                exp_regression_data
            )
        )

    def test_intents_lists(self) -> None:
        """

        :return:
        :rtype:
        """
        exp_training_intents_list = [
            "TestTrainIntent"
        ]
        ExternalDatasets.set_training_data()
        self.assertListEqual(
            ExternalDatasets.training_intents_list(),
            exp_training_intents_list
        )

        exp_testing_intents_list = [
            "TestTestIntent"
        ]
        ExternalDatasets.set_testing_data()
        self.assertListEqual(
            ExternalDatasets.testing_intents_list(),
            exp_testing_intents_list
        )

        exp_evaluation_intents_list = [
            "TestEvaluateIntent"
        ]
        ExternalDatasets.set_evaluation_data()
        self.assertListEqual(
            ExternalDatasets.evaluation_intents_list(),
            exp_evaluation_intents_list
        )

        exp_regression_intents_list = [
            "TestRegressIntent"
        ]
        ExternalDatasets.set_regression_data()
        self.assertListEqual(
            ExternalDatasets.regression_intents_list(),
            exp_regression_intents_list
        )

    def test_intents_sets(self) -> None:
        """

        :return:
        :rtype:
        """
        exp_training_intents_set = {
            "TestTrainIntent"
        }
        ExternalDatasets.set_training_data()
        self.assertSetEqual(
            ExternalDatasets.training_intents_set(),
            exp_training_intents_set
        )

        exp_testing_intents_set = {
            "TestTestIntent"
        }
        ExternalDatasets.set_testing_data()
        self.assertSetEqual(
            ExternalDatasets.testing_intents_set(),
            exp_testing_intents_set
        )

        exp_evaluation_intents_set = {
            "TestEvaluateIntent"
        }
        ExternalDatasets.set_evaluation_data()
        self.assertSetEqual(
            ExternalDatasets.evaluation_intents_set(),
            exp_evaluation_intents_set
        )

        exp_regression_intents_set = {
            "TestRegressIntent"
        }
        ExternalDatasets.set_regression_data()
        self.assertSetEqual(
            ExternalDatasets.regression_intents_set(),
            exp_regression_intents_set
        )

    def test_intents_counts(self) -> None:
        """

        :return:
        :rtype:
        """
        exp_training_intents_counts = pd.DataFrame({
            "text": [
                "testTrain"
            ],
            "intent": [
                "TestTrainIntent"
            ]
        }).intent.value_counts()
        ExternalDatasets.set_training_data()
        print(ExternalDatasets.training_intents_counts())
        print(exp_training_intents_counts)
        self.assertTrue(
            ExternalDatasets.training_intents_counts().equals(
                exp_training_intents_counts
            )
        )

        exp_testing_intents_counts = pd.DataFrame({
            "text": [
                "testTest"
            ],
            "intent": [
                "TestTestIntent"
            ]
        }).intent.value_counts()
        ExternalDatasets.set_testing_data()
        self.assertTrue(
            ExternalDatasets.testing_intents_counts().equals(
                exp_testing_intents_counts
            )
        )

        exp_evaluation_intents_counts = pd.DataFrame({
            "text": [
                "testEvaluate"
            ],
            "intent": [
                "TestEvaluateIntent"
            ]
        }).intent.value_counts()
        ExternalDatasets.set_evaluation_data()
        self.assertTrue(
            ExternalDatasets.evaluation_intents_counts().equals(
                exp_evaluation_intents_counts
            )
        )

        exp_regression_intents_counts = pd.DataFrame({
            "text": [
                "testRegress"
            ],
            "intent": [
                "TestRegressIntent"
            ]
        }).intent.value_counts()
        ExternalDatasets.set_regression_data()
        self.assertTrue(
            ExternalDatasets.regression_intents_counts().equals(
                exp_regression_intents_counts
            )
        )

    def test_all_intents(self) -> None:
        """

        :return:
        :rtype:
        """
        exp_all_intents = sorted([
            "TestTrainIntent",
            "TestTestIntent",
            "TestEvaluateIntent",
            "TestRegressIntent"
        ])
        ExternalDatasets.set_training_data()
        ExternalDatasets.set_testing_data()
        ExternalDatasets.set_evaluation_data()
        ExternalDatasets.set_regression_data()
        self.assertListEqual(
            ExternalDatasets.all_intents(),
            exp_all_intents
        )

    def test_check_intents_intersect(self) -> None:
        """

        :return:
        :rtype:
        """
        # training vs. testing
        os.remove(WoodgateSettings.get_training_path())
        os.remove(WoodgateSettings.get_testing_path())
        os.remove(WoodgateSettings.get_evaluation_path())
        os.remove(WoodgateSettings.get_regression_path())

        os.makedirs(
            os.path.dirname(
                WoodgateSettings.get_training_path()
            ),
            exist_ok=True
        )
        with open(
                WoodgateSettings.get_training_path(), "w+"
        ) as file:
            file.writelines([
                "text,intent\n",
                "test,TestIntent\n"
            ])

        os.makedirs(
            os.path.dirname(
                WoodgateSettings.get_testing_path()
            ),
            exist_ok=True
        )
        with open(
                WoodgateSettings.get_testing_path(), "w+"
        ) as file:
            file.writelines([
                "text,intent\n",
                "test,TestIntent\n"
            ])

        os.makedirs(
            os.path.dirname(
                WoodgateSettings.get_evaluation_path()
            ),
            exist_ok=True
        )
        with open(
                WoodgateSettings.get_evaluation_path(), "w+"
        ) as file:
            file.writelines([
                "text,intent\n",
                "test,TestIntent\n"
            ])

        os.makedirs(
            os.path.dirname(
                WoodgateSettings.get_regression_path()
            ),
            exist_ok=True
        )
        with open(
                WoodgateSettings.get_regression_path(), "w+"
        ) as file:
            file.writelines([
                "text,intent\n",
                "test,TestIntent\n"
            ])
        ExternalDatasets.set_training_data()
        ExternalDatasets.set_testing_data()
        ExternalDatasets.set_evaluation_data()
        ExternalDatasets.set_regression_data()
        ExternalDatasets.assert_intents_intersect()

    def test_intents_do_not_intersect(self) -> None:
        """

        :return:
        :rtype:
        """
        ExternalDatasets.set_training_data()
        ExternalDatasets.set_testing_data()
        ExternalDatasets.set_evaluation_data()
        ExternalDatasets.set_regression_data()
        self.assertRaises(
            ValueError,
            ExternalDatasets.assert_intents_intersect,
        )

        # train vs. evaluation
        os.remove(WoodgateSettings.get_training_path())
        os.remove(WoodgateSettings.get_testing_path())
        os.remove(WoodgateSettings.get_evaluation_path())
        os.remove(WoodgateSettings.get_regression_path())

        os.makedirs(
            os.path.dirname(
                WoodgateSettings.get_training_path()
            ),
            exist_ok=True
        )
        with open(
                WoodgateSettings.get_training_path(), "w+"
        ) as file:
            file.writelines([
                "text,intent\n",
                "test,TestIntent\n"
            ])

        os.makedirs(
            os.path.dirname(
                WoodgateSettings.get_testing_path()
            ),
            exist_ok=True
        )
        with open(
                WoodgateSettings.get_testing_path(), "w+"
        ) as file:
            file.writelines([
                "text,intent\n",
                "test,TestIntent\n"
            ])

        os.makedirs(
            os.path.dirname(
                WoodgateSettings.get_evaluation_path()
            ),
            exist_ok=True
        )
        with open(
                WoodgateSettings.get_evaluation_path(), "w+"
        ) as file:
            file.writelines([
                "text,intent\n",
                "test,TestEvaluationIntent\n"
            ])

        os.makedirs(
            os.path.dirname(
                WoodgateSettings.get_regression_path()
            ),
            exist_ok=True
        )
        with open(
                WoodgateSettings.get_regression_path(), "w+"
        ) as file:
            file.writelines([
                "text,intent\n",
                "test,TestIntent\n"
            ])
        ExternalDatasets.set_training_data()
        ExternalDatasets.set_testing_data()
        ExternalDatasets.set_evaluation_data()
        ExternalDatasets.set_regression_data()
        self.assertRaises(
            ValueError,
            ExternalDatasets.assert_intents_intersect,
        )

        # train vs. regression
        os.remove(WoodgateSettings.get_training_path())
        os.remove(WoodgateSettings.get_testing_path())
        os.remove(WoodgateSettings.get_evaluation_path())
        os.remove(WoodgateSettings.get_regression_path())

        os.makedirs(
            os.path.dirname(
                WoodgateSettings.get_training_path()
            ),
            exist_ok=True
        )
        with open(
                WoodgateSettings.get_training_path(), "w+"
        ) as file:
            file.writelines([
                "text,intent\n",
                "test,TestIntent\n"
            ])

        os.makedirs(
            os.path.dirname(
                WoodgateSettings.get_testing_path()
            ),
            exist_ok=True
        )
        with open(
                WoodgateSettings.get_testing_path(), "w+"
        ) as file:
            file.writelines([
                "text,intent\n",
                "test,TestIntent\n"
            ])

        os.makedirs(
            os.path.dirname(
                WoodgateSettings.get_evaluation_path()
            ),
            exist_ok=True
        )
        with open(
                WoodgateSettings.get_evaluation_path(), "w+"
        ) as file:
            file.writelines([
                "text,intent\n",
                "test,TestIntent\n"
            ])

        os.makedirs(
            os.path.dirname(
                WoodgateSettings.get_regression_path()
            ),
            exist_ok=True
        )
        with open(
                WoodgateSettings.get_regression_path(), "w+"
        ) as file:
            file.writelines([
                "text,intent\n",
                "test,TestRegressionIntent\n"
            ])
        ExternalDatasets.set_training_data()
        ExternalDatasets.set_testing_data()
        ExternalDatasets.set_evaluation_data()
        ExternalDatasets.set_regression_data()
        self.assertRaises(
            ValueError,
            ExternalDatasets.assert_intents_intersect,
        )

    def test_intents_dict(self) -> None:
        """

        :return:
        :rtype:
        """

        ExternalDatasets.set_training_data()
        ExternalDatasets.set_testing_data()
        ExternalDatasets.set_evaluation_data()
        ExternalDatasets.set_regression_data()

        exp_intents_dict = {
            "intents": sorted([
                "TestTrainIntent",
                "TestTestIntent",
                "TestEvaluateIntent",
                "TestRegressIntent"
            ]),
            "training_set": {
                "intents": ["TestTrainIntent"],
                "counts": pd.DataFrame({
                    "text": [
                        "testTrain"
                    ],
                    "intent": [
                        "TestTrainIntent"
                    ]
                }).intent.value_counts().to_dict()
            },
            "testing_set": {
                "intents": ["TestTestIntent"],
                "counts": pd.DataFrame({
                    "text": [
                        "testTest"
                    ],
                    "intent": [
                        "TestTestIntent"
                    ]
                }).intent.value_counts().to_dict()
            },
            "evaluation_set": {
                "intents": ["TestEvaluateIntent"],
                "counts": pd.DataFrame({
                    "text": [
                        "testEvaluate"
                    ],
                    "intent": [
                        "TestEvaluateIntent"
                    ]
                }).intent.value_counts().to_dict()
            },
            "regression_set": {
                "intents": ["TestRegressIntent"],
                "counts": pd.DataFrame({
                    "text": [
                        "testRegress"
                    ],
                    "intent": [
                        "TestRegressIntent"
                    ]
                }).intent.value_counts().to_dict()
            }
        }

        self.assertDictEqual(
            ExternalDatasets.intents_dict(),
            exp_intents_dict
        )

    def test_create_data_json(self) -> None:
        """

        :return:
        :rtype:
        """
        ExternalDatasets.set_training_data()
        ExternalDatasets.set_testing_data()
        ExternalDatasets.set_evaluation_data()
        ExternalDatasets.set_regression_data()
        ExternalDatasets.create_intents_data_json()
        exp_intents_dict = {
            "intents": sorted([
                "TestTrainIntent",
                "TestTestIntent",
                "TestEvaluateIntent",
                "TestRegressIntent"
            ]),
            "training_set": {
                "intents": ["TestTrainIntent"],
                "counts": pd.DataFrame({
                    "text": [
                        "testTrain"
                    ],
                    "intent": [
                        "TestTrainIntent"
                    ]
                }).intent.value_counts().to_dict()
            },
            "testing_set": {
                "intents": ["TestTestIntent"],
                "counts": pd.DataFrame({
                    "text": [
                        "testTest"
                    ],
                    "intent": [
                        "TestTestIntent"
                    ]
                }).intent.value_counts().to_dict()
            },
            "evaluation_set": {
                "intents": ["TestEvaluateIntent"],
                "counts": pd.DataFrame({
                    "text": [
                        "testEvaluate"
                    ],
                    "intent": [
                        "TestEvaluateIntent"
                    ]
                }).intent.value_counts().to_dict()
            },
            "regression_set": {
                "intents": ["TestRegressIntent"],
                "counts": pd.DataFrame({
                    "text": [
                        "testRegress"
                    ],
                    "intent": [
                        "TestRegressIntent"
                    ]
                }).intent.value_counts().to_dict()
            }
        }
        with open(
                os.path.join(
                    WoodgateSettings.datasets_summary_dir,
                    "intentsData.json"
                )
        ) as file:
            loaded_dict = json.loads(file.read())
            print(loaded_dict)
            print(exp_intents_dict)
            self.assertDictEqual(loaded_dict, exp_intents_dict)

    def test_create_bar_plots(self) -> None:
        """

        :return:
        :rtype:
        """
        ExternalDatasets.set_training_data()
        ExternalDatasets.set_testing_data()
        ExternalDatasets.set_evaluation_data()
        ExternalDatasets.set_regression_data()
        ExternalDatasets.create_intents_bar_plots()
        exp_bar_plot_paths_list = [
            os.path.join(
                WoodgateSettings.datasets_summary_dir,
                "intents_bar_plot_training.png"
            ),
            os.path.join(
                WoodgateSettings.datasets_summary_dir,
                "intents_bar_plot_testing.png"
            ),
            os.path.join(
                WoodgateSettings.datasets_summary_dir,
                "intents_bar_plot_evaluation.png"
            ),
            os.path.join(
                WoodgateSettings.datasets_summary_dir,
                "intents_bar_plot_regression.png"
            ),
        ]

        for path in exp_bar_plot_paths_list:
            self.assertTrue(os.path.isfile(path))

    def test_create_venn_diagram(self) -> None:
        """

        :return:
        :rtype:
        """
        ExternalDatasets.set_training_data()
        ExternalDatasets.set_testing_data()
        ExternalDatasets.set_evaluation_data()
        ExternalDatasets.set_regression_data()
        ExternalDatasets.create_intents_venn_diagrams()
        exp_venn_diagrams_paths_list = [
            os.path.join(
                WoodgateSettings.datasets_summary_dir,
                "intents_venn_training_evaluation.png"
            ),
            os.path.join(
                WoodgateSettings.datasets_summary_dir,
                "intents_venn_training_testing.png"
            ),
            os.path.join(
                WoodgateSettings.datasets_summary_dir,
                "intents_venn_training_regression.png"
            ),
            os.path.join(
                WoodgateSettings.datasets_summary_dir,
                "intents_venn_evaluation_testing.png"
            ),
            os.path.join(
                WoodgateSettings.datasets_summary_dir,
                "intents_venn_evaluation_regression.png"
            ),
            os.path.join(
                WoodgateSettings.datasets_summary_dir,
                "intents_venn_testing_regression.png"
            ),
        ]

        for path in exp_venn_diagrams_paths_list:
            self.assertTrue(os.path.isfile(path))


if __name__ == '__main__':
    unittest.main()
