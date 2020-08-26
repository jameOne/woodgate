"""
woodgate_settings.py - The woodgate_settings.py module contains
the WoodgateSettings class definition.
"""
import os
import uuid
import datetime


class Architecture:
    """
    Architecture
    """
    def __init__(
            self,
            clf_out_dropout_rate: float,
            clf_out_activation: str,
            logits_dropout_rate: float,
            logits_activation: str
    ):
        """

        :param clf_out_dropout_rate:
        :type clf_out_dropout_rate:
        :param clf_out_activation:
        :type clf_out_activation:
        :param logits_dropout_rate:
        :type logits_dropout_rate:
        :param logits_activation:
        :type logits_activation:
        """
        #: The `clf_out_dropout_rate` attribute represents one
        #: of two (1 / 2) dropout rates which may be customized.
        #: Typically this value should be around `0.5`. This
        #: attribute is set via the `CLF_OUT_DROPOUT_RATE`
        #: environment variable. If the `CLF_OUT_DROPOUT_RATE`
        #: environment variable is not set, then the
        #: `clf_out_dropout_rate` defaults to `0.5`.
        self.clf_out_dropout_rate: float = clf_out_dropout_rate

        #: The `clf_out_activation` attribute represents one
        #: of two (1 / 2) activation functions which may be
        #: customized. This attribute is set via the
        #: `CLF_OUT_ACTIVATION` environment variable. If the
        #: `CLF_OUT_ACTIVATION` environment variable is not set,
        #: then the `clf_out_activation` defaults to `tanh`.
        self.clf_out_activation: str = clf_out_activation

        #: The `logits_dropout_rate` attribute represents two
        #: of two (2 / 2) dropout rates which may be customized.
        #: Typically this value should be around `0.5`. This
        #: attribute is set via the `LOGITS_DROPOUT_RATE`
        #: environment variable. If the `LOGITS_DROPOUT_RATE`
        #: environment variable is not set, then the
        #: `logits_dropout_rate` defaults to `0.5`.
        self.logits_dropout_rate: float = logits_dropout_rate

        #: The `logits_activation` attribute represents two
        #: of two (2 / 2) activation functions which may be
        #: customized. This attribute is set via the
        #: `LOGITS_ACTIVATION` environment variable. If the
        #: `LOGITS_ACTIVATION` environment variable is not set,
        #: then the `logits_activation` defaults to `tanh`.
        self.logits_activation: str = logits_activation


class Model:
    """
    Model
    """

    def __init__(
            self,
            model_name: str,
            model_uuid: str = ""
    ):
        #: The `model_name` attribute represents the name given
        #: to the machine learning evaluator. This attribute is
        #: set via the `MODEL_NAME` environment variable. If the
        #: `MODEL_NAME` environment variable is not set, the
        #: `model_name` attribute is set to a random (v4) UUID by
        #: default.
        self.model_name: str = model_name

        #: The `model_uuid` attribute represents the name given
        #: to the machine learning evaluator. The `model_uuid`
        #: attribute is set to a random (v4) UUID. This attribute
        #: cannot be manually set.
        try:
            model_uuid = uuid.UUID(model_uuid, version=4)
            self.model_uuid: str = str(model_uuid)
        except ValueError:
            # TODO - This should be logged.
            self.model_uuid: str = str(uuid.uuid4())


class Build:
    """
    Build
    """
    #: The `build_version` attribute represents the specific
    #: version of the evaluator build_history. This attribute
    #: is set via the `BUILD_VERSION` environment variable. If
    #: the `BUILD_VERSION` environment variable is not set, the
    #: `build_version` attribute is set to a string formatted
    #: time ("%Y%m%d%H%M%S") by default.
    build_version: str = \
        datetime.datetime.now().strftime("%Y%m%d%H%M%S")


class FileSystem:
    """
    FileSystem class encapsulates logic related to
    reading the required settings/configuration from the
    environment i.e. via environment variables.
    """

    def __init__(
            self,
            model: Model,
            build: Build
    ):
        """

        :param model:
        :type model:
        :param build:
        :type build:
        """

        #: The `woodgate_base_dir` attribute represents the path
        #: to a directory on the host file system from which
        #: many of the other file paths are constructed. This
        #: directory is also where files generated or downloaded
        #: by the process are stored. This attribute is set via
        #: the `WOODGATE_BASE_DIR` environment variable. If the
        #: `WOODGATE_BASE_DIR` environment variable is not set,
        #: then the `woodgate_base_dir` attribute is set to
        #: `~/woodgate`. The program will attempt to create
        #: `WOODGATE_BASE_DIR` if it does not already exist.
        self.woodgate_base_dir: str = os.getenv(
            "WOODGATE_BASE_DIR",
            os.path.join(
                os.path.expanduser("~"),
                "woodgate"
            )
        )

        #: The `model_dir` attribute represents the path
        #: to a directory on the host file system where the
        #: directory learning evaluator will be stored. The
        #: evaluator build_history directory should be a child
        #: directory of the build_history directory in which the
        #: program will store the tuned evaluator after training.
        #: This attribute is set via the `MODEL_DIR` environment
        #: variable. If the `MODEL_DIR` environment variable is
        #: not set, then the `model_dir` attribute will default
        #: to `$BUILD_DIR/$MODEL_NAME`. The program will attempt
        #: to create `MODEL_DIR` if it does not already exist.
        self.model_dir: str = os.getenv(
            "MODEL_DIR",
            os.path.join(
                self.woodgate_base_dir,
                model.model_uuid
            )
        )

        #: The `temp_dir` attribute represents the path
        #: to a directory on the host file system where
        #: temporary files will be stored. The
        #: directory should be a child directory of the
        #: `woodgate_base_dir` directory.
        #: This attribute is set via the `TEMP_DIR` environment
        #: variable. If the `TEMP_DIR` environment variable is
        #: not set, then the `temp_dir` attribute will default
        #: to `$WOODGATE_BASE_DIR/temp`. The program will attempt
        #: to create `TEMP_DIR` if it does not already exist.
        self.temp_dir: str = os.getenv(
            "TEMP_DIR",
            os.path.join(
                self.woodgate_base_dir,
                "temp"
            )
        )

        #: The `data_dir` attribute represents the path to a
        #: directory on the host file system where data files
        #: are stored. This attribute is set via the `DATA_DIR`
        #: environment variable. The data directory should be
        #: the parent directory of `training_dir`
        #: (`TRAINING_DIR`), `testing_dir` (`TESTING_DIR`),
        #: `evaluation_dir` (`EVALUATION_DIR`), `validation_dir`
        #: (`VALIDATION_DIR`) and `regression_dir`
        #: (`REGRESSION_DIR`). If the `DATA_DIR` is not set,
        #: then the `data_dir` attribute is set to
        #: `$WOODGATE_BASE_DIR/data` by default. The program
        #: will attempt to create `DATA_DIR` if it does not
        #: already exist.
        self.data_dir: str = os.getenv(
            "DATA_DIR",
            os.path.join(
                self.woodgate_base_dir,
                "data"
            )
        )

        #: The `build_dir` attribute represents the path to a
        #: directory on the host file system where the versioned
        #: build_history output will be stored. Versioned
        #: build_history output simply means that individual
        #: builds are namespaced by version within the
        #: `$OUTPUT_DIR`. This attribute is set via the
        #: `BUILD_DIR` environment variable. For example, if the
        #: `output_dir` attribute takes the value `$OUTPUT_DIR`
        #: then the `build_dir` attribute should look like
        #: `$OUTPUT_DIR/%Y%m%d-%H%M%s` for a string formatted
        #: date `%Y%m%d-%H%M%s`. The program will attempt to
        #: create `BUILD_DIR` if it does not already exist.
        self.build_dir: str = os.getenv(
            "BUILD_DIR",
            os.path.join(
                self.model_dir,
                build.build_version
            )
        )

        #: The `log_dir` attribute represents the path to a
        #: directory on the host file system where the
        #: log data will be stored. The log data directory should
        #: be a child directory of the build_history output
        #: directory in which the program will store the data
        #: retrieved for logging the build_history process. This
        #: attribute is set via the `LOG_DIR` environment
        #: variable. If the `LOG_DIR` environment variable is not
        #: set, then the `log_dir` attribute will default to
        #: `$OUTPUT_DIR/log`. The program will attempt to create
        #: `LOG_DIR` if it does not already exist.
        self.log_dir: str = os.getenv(
            "LOG_DIR",
            os.path.join(
                self.build_dir,
                "log"
            )
        )

        #: The `log_file` attribute represents the base name of
        #: the `log_path` attribute. The log file should
        #: therefore reside in the `log_dir` by definition.
        #: This file should have a CSV file
        #: (having a `.csv` file extension). This
        #: attribute is set via the `REGRESSION_FILE` environment
        #: variable. If the `LOG_FILE` environment variable is
        #: not set, then the `log_file` will default to
        #: `$BUILD_VERSION.log`.
        self.log_file: str = os.getenv(
            "LOG_FILE",
            f"{build.build_version}.log"
        )

        #: The `training_dir` attribute represents the path to a
        #: directory on the host file system where the
        #: training data will be stored. The training data
        #: directory should be a child directory of the evaluator
        #: data directory in which the program will store the
        #: data retrieved for training the learning evaluator.
        #: This attribute is set via the `TRAINING_DIR`
        #: environment variable. If the `TRAINING_DIR`
        #: environment variable is not set, then the
        #: `training_dir` attribute will default to
        #: `$MODEL_DATA_DIR/train`. The program will
        #: attempt to create `TRAINING_DIR` if it does not
        #: already exist.
        self.training_dir: str = os.getenv(
            "TRAINING_DIR",
            os.path.join(
                self.build_dir,
                "training_data"
            )
        )

        #: The `training_file` attribute represents the base
        #: name of the `training_path` attribute. The training
        #: file should therefore reside in the `training_dir` by
        #: definition. This file should have a CSV file (having a
        #: `.csv` file extension). This attribute is set via the
        #: `TRAINING_FILE` environment variable. The training
        #: file should contain at least two (2) columns; one (1)
        #: for text having title `text_column_title` (set via
        #: `TEXT_COLUMN_TITLE` environment variable) and one (1)
        #: for label having title `label_column_title`
        #: (set via `LABEL_COLUMN_TITLE` environment variable).
        self.training_file: str = os.getenv(
            "TRAINING_FILE",
            "train.csv"
        )

        #: The `testing_dir` attribute represents the path to a
        #: directory on the host file system where the
        #: testing data will be stored. The testing data
        #: directory should be a child directory of the evaluator
        #: data directory in which the program will store the
        #: data retrieved for testing the learning evaluator.
        #: This attribute is set via the `TESTING_DIR`
        #: environment variable. If the `TESTING_DIR` environment
        #: variable is not set, then the `testing_dir` attribute
        #: will default to `$MODEL_DATA_DIR/test`. The program
        #: will attempt to create `TESTING_DIR` if it does not
        #: already exist.
        self.testing_dir: str = os.getenv(
            "TESTING_DIR",
            os.path.join(
                self.build_dir,
                "testing_data"
            )
        )

        #: The `testing_file` attribute represents the base name
        #: of the `testing_path` attribute. The testing file
        #: should therefore reside in the `testing_dir` by
        #: definition. This file should have a CSV file (having a
        #: `.csv` file extension). This attribute is set via the
        #: `TESTING_FILE` environment variable. The testing
        #: file should contain at least two (2) columns; one (1)
        #: for text having title `text_column_title`
        #: (set via `TEXT_COLUMN_TITLE` environment variable) and
        #: one (1) for label having title `label_column_title`
        #: (set via `LABEL_COLUMN_TITLE` environment variable).
        self.testing_file: str = os.getenv(
            "TESTING_FILE",
            "test.csv"
        )

        #: The `evaluation_dir` attribute represents the path to
        #: a directory on the host file system where the
        #: evaluation data will be stored. The evaluation data
        #: directory should be a child directory of the evaluator
        #: data directory in which the program will store the
        #: dats retrieved for validating the learning evaluator.
        #: This attribute is set via the `EVALUATION_DIR`
        #: environment variable. If the `EVALUATION_DIR`
        #: environment variable is not set, then the
        #: `evaluation_dir` attribute will default to
        #: `$MODEL_DATA_DIR/evaluate`. The program will attempt
        #: to create `EVALUATION_DIR` if it does not already
        #: exist.
        self.evaluation_dir: str = os.getenv(
            "EVALUATION_DIR",
            os.path.join(
                self.build_dir,
                "evaluation_data"
            )
        )

        #: The `evaluation_file` attribute represents the base
        #: name of the `evaluation_path` attribute. The
        #: evaluation file should therefore reside in the
        #: `evaluation_dir` by definition. This file should have
        #: a CSV file (having a `.csv` file extension). This
        #: attribute is set via the `EVALUATION_FILE` environment
        #: variable. The evaluation file should contain at
        #: least two (2) columns; one (1) for text having title
        #: `text_column_title` (set via `TEXT_COLUMN_TITLE`
        #: environment variable) and one (1) for label having
        #: title `label_column_title` (set via
        #: `LABEL_COLUMN_TITLE` environment variable).
        self.evaluation_file: str = os.getenv(
            "EVALUATION_FILE",
            "evaluate.csv"
        )

        #: The `regression_dir` attribute represents the path to
        #: a directory on the host file system where the
        #: regression data will be stored. The regression data
        #: directory should be a child directory of the evaluator
        #: data directory in which the program will store the
        #: data retrieved for validating the learning evaluator.
        #: This attribute is set via the `REGRESSION_DIR`
        #: environment variable. If the `REGRESSION_DIR`
        #: environment variable is not set, then the
        #: `regression_dir` attribute will default to
        #: `$MODEL_DATA_DIR/regress`. The program will attempt
        #: to create `REGRESSION_DIR` if it does not already
        #: exist.
        self.regression_dir: str = os.getenv(
            "REGRESSION_DIR",
            os.path.join(
                self.build_dir,
                "regression_data"
            )
        )

        #: The `regression_file` attribute represents the base
        #: name of the `regression_path` attribute. The
        #: regression file should therefore reside in the
        #: `regression_dir` by definition. This file should have
        #: a CSV file (having a `.csv` file extension). This
        #: attribute is set via the `REGRESSION_FILE`
        #: environment variable. The regression file should
        #: contain at least two (2) columns; one (1) for text
        #: having title `text_column_title`
        #: (set via `TEXT_COLUMN_TITLE` environment variable)
        #: and one (1) for label having title `label_column_title`
        #: (set via `LABEL_COLUMN_TITLE` environment variable).
        self.regression_file: str = os.getenv(
            "REGRESSION_FILE",
            "regress.csv"
        )

        #: The `datasets_summary_dir` attribute represents a
        #: directory on the host's file system. This is where the
        #: summary files generated by the
        #: `woodgate.tuning.external_datasets.create_*` methods
        #: are stored. The program will attempt to
        #: create `DATASETS_SUMMARY_DIR` if it does not already
        #: exist.
        self.datasets_summary_dir: str = os.getenv(
            "DATASETS_SUMMARY_DIR",
            os.path.join(
                self.build_dir,
                "datasets_summary"
            )
        )

        #: The `build_summary_dir` attribute represents a
        #: directory on the host's file system. This is where the
        #: summary files generated by the
        #: `woodgate.build_history.build_summary.create_*` methods
        #: are stored. The program will attempt to
        #: create `BUILD_SUMMARY_DIR` if it does not already
        #: exist.
        self.build_summary_dir: str = os.getenv(
            "BUILD_SUMMARY_DIR",
            os.path.join(
                self.build_dir,
                "build_summary"
            )
        )

        #: The `evaluation_summary_dir` attribute represents a
        #: directory on the host's file system. This is where the
        #: summary files generated by the
        #: `woodgate.evaluator.evaluation_summary.create_*`
        #: methods are stored. The program will attempt to
        #: create `EVALUATION_SUMMARY_DIR` if it does not already
        #: exist.
        self.evaluation_summary_dir: str = os.getenv(
            "EVALUATION_SUMMARY_DIR",
            os.path.join(
                self.build_dir,
                "evaluation_summary"
            )
        )

        #: The `bert_dir` attribute represents a directory on the
        #: host file system containing the BERT transfer evaluator
        #: and associated files. This attribute is set via the
        #: `BERT_DIR` environment variable. If the `BERT_DIR`
        #: environment variable is not set, then the `bert_dir`
        #: attribute defaults to `$WOODGATE_BASE_DIR/bert`.
        #: The program will attempt to create `BERT_DIR` if it
        #: does not already exist.
        self.bert_dir: str = os.getenv(
            "BERT_DIR",
            os.path.join(
                self.woodgate_base_dir,
                "bert"
            )
        )

        #: The `bert_config_file` attribute represents the name
        #: of the BERT configuration JSON file. This attribute is
        #: set via the `BERT_CONFIG_FILE` environment variable.
        #: If the `BERT_CONFIG_FILE` environment variable is
        #: not set, then the `bert_config_file` defaults to
        #: `bert_config.json`.
        self.bert_config_file: str = os.getenv(
            "BERT_CONFIG_FILE",
            "bert_config.json"
        )

        #: The `bert_model_file` attribute represents the name
        #: of the BERT evaluator (`.ckpt` file extension). This
        #: attribute is set via the `BERT_MODEL_FILE` environment
        #: variable. If the `BERT_MODEL_FILE` environment
        #: variable is not set, then the `bert_model_file`
        #: defaults to `bert_model.ckpt`.
        self.bert_model_file: str = os.getenv(
            "BERT_MODEL_FILE",
            "bert_model.ckpt"
        )

        #: The `bert_vocab_file` attribute represents the
        #: name of the BERT evaluator's vocabulary file
        #: (`.txt` file extension). This attribute is set via
        #: the `BERT_VOCAB_FILE` environment variable. If the
        #: `BERT_VOCAB_FILE` environment variable is not set,
        #: then the `bert_vocab_file` defaults to `vocab.txt`.
        self.bert_vocab_file: str = os.getenv(
            "BERT_VOCAB_FILE",
            "vocab.txt"
        )

    def configure(self) -> None:
        """This method will iterate over the instance's
        attributes and pass the

        :return: None
        :rtype: NoneType
        """
        for attr, val in self.__dict__.items():
            if attr[-4:] == "_dir":
                os.makedirs(val, exist_ok=True)

        return None

    def get_log_path(self) -> str:
        """The `get_log_path` method returns the full path on
        the host file system pointing to `log_file`.

        :return:
        :rtype:
        """
        return os.path.join(self.log_dir, self.log_file)

    def get_training_path(self) -> str:
        """The `get_training_path` method returns the full path
        on the host file system pointing to `training_file`.

        :return: Path to `self.training_file`
        :rtype: str
        """

        return os.path.join(self.training_dir, self.training_file)

    def get_testing_path(self) -> str:
        """The `get_testing_path` method returns the full path
        on the host file system pointing to `testing_file`.

        :return: Path to `self.testing_file`
        :rtype: str
        """

        return os.path.join(self.testing_dir, self.testing_file)

    def get_regression_path(self) -> str:
        """The `get_regression_path` method returns the full path
        on the host file system pointing to `regression_file`.

        :return: Path to `self.regression_file`
        :rtype: str
        """

        return os.path.join(
            self.regression_dir, self.regression_file)

    def get_evaluation_path(self) -> str:
        """The `get_evaluation_path` method returns the full path
        on the host file system pointing to `evaluation_file`.

        :return: Path to `self.evaluation_file`
        :rtype: str
        """

        return os.path.join(
            self.evaluation_dir, self.evaluation_file)

    def get_bert_config_path(self) -> str:
        """The `get_bert_config_path` method returns the full
        path on the host file system pointing to
        `bert_config_file`.

        :return: Path to `self.bert_config_file`
        :rtype: str
        """
        return os.path.join(self.bert_dir, self.bert_config_file)

    def get_bert_model_path(self) -> str:
        """The `get_bert_model_path` method returns the full
        path on the host file system pointing to
        `bert_config_file`.

        :return: Path to `self.bert_config_file`
        :rtype: str
        """

        return os.path.join(self.bert_dir, self.bert_model_file)

    def get_bert_vocab_path(self):
        """The `get_bert_vocab_path` method returns the full
        path on the host file system pointing to
        `bert_vocab_file`.

        :return: Path to `self.bert_vocab_file`
        :rtype: str
        """
        return os.path.join(self.bert_dir, self.bert_vocab_file)
