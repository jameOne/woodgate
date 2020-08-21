"""
woodgate_settings.py - The woodgate_settings.py module contains
the WoodgateSettings class definition.
"""
import os
import ast
import uuid
import datetime
from typing import List, Any


class WoodgateSettings:
    """
    WoodgateSettings class encapsulates logic related to
    reading the required settings/configuration from the
    environment i.e. via environment variables.
    """

    #: The `model_name` attribute represents the name given to
    #: the machine learning model. This attribute is set via
    #: the `MODEL_NAME` environment variable. If the
    #: `MODEL_NAME` environment variable is not set, the
    #: `model_name` attribute is set to a random (v4) UUID by
    #: default.
    model_name: str = os.getenv(
        "MODEL_NAME",
        str(uuid.uuid4())
    )

    #: The `build_version` attribute represents the specific
    #: version of the model build. This attribute is set via
    #: the `BUILD_VERSION` environment variable. If the
    #: `BUILD_VERSION` environment variable is not set, the
    #: `build_version` attribute is set to a string formatted
    #: time ("%Y%m%d-%H%M%s") by default.
    build_version: str = os.getenv(
        "BUILD_VERSION",
        datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    )

    #: The `create_dataset_visuals` attribute represents a
    #: signal variable that is used to decide whether visuals
    #: should be generated along with a text summary of the
    #: training, testing, validation, evaluation, and
    #: regression datasets. This attribute is set via the
    #: `CREATE_DATASET_VISUALS` environment variable. If the
    #: `CREATE_DATASET_VISUALS` environment variable is not
    #: set, then the create_dataset_visuals attribute is set
    #: to `1` by default signaling the program to generate
    #: visuals. All values except `CREATE_DATASET_VISUALS=0`
    #: signal the program to generate dataset visuals.
    create_dataset_visuals: int = int(
        os.getenv("CREATE_DATASET_VISUALS", "1")
    )

    #: The `create_build_visuals` attribute represents a
    #: signal variable that is used to decide whether visuals
    #: should be generated along with a text summary of the
    #: build history. This attribute is set via the
    #: `CREATE_BUILD_VISUALS` environment variable. If the
    #: `CREATE_BUILD_VISUALS` environment variable is not set,
    #: then the `create_build_visuals` attribute is set to `1`
    #: by  default signaling the program to generate visuals.
    #: All  values except `CREATE_DATASET_VISUALS=0` signal
    #: the program to generate build visuals.
    create_build_visuals: int = int(
        os.getenv("CREATE_BUILD_VISUALS", "1")
    )

    #: The `create_evaluation_visuals` attribute represents a
    #: signal variable that is used to decide whether visuals
    #: should be generated along with a text summary of the
    #: evaluation history. This attribute is set via the
    #: `CREATE_EVALUATION_VISUALS` environment variable. If the
    #: `CREATE_EVALUATION_VISUALS` environment variable is not
    #: set, then the `create_evaluation_visuals` attribute is set
    #: to `1` by  default signaling the program to generate
    #: visuals.  All  values except `CREATE_DATASET_VISUALS=0`
    #: signal the program to generate evaluation visuals.
    create_evaluation_visuals: int = int(
        os.getenv("CREATE_EVALUATION_VISUALS", "1")
    )

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
    woodgate_base_dir: str = os.getenv(
        "WOODGATE_BASE_DIR",
        os.path.join(
            os.path.expanduser("~"),
            "woodgate"
        )
    )

    #: The `data_dir` attribute represents the path to a
    #: directory on the host file system where data files
    #: are stored. This attribute is set via the `DATA_DIR`
    #: environment variable. The data directory should be the
    #: parent directory of `training_dir` (`TRAINING_DIR`),
    #: `testing_dir` (`TESTING_DIR`), `evaluation_dir`
    #: (`EVALUATION_DIR`), `validation_dir` (`VALIDATION_DIR`)
    #: and `regression_dir` (`REGRESSION_DIR`).
    #: If the `DATA_DIR` is not set, then the `data_dir`
    #: attribute is set to `$WOODGATE_BASE_DIR/data` by
    #: default. The program will attempt to create `DATA_DIR`
    #: if it does not already exist.
    data_dir: str = os.getenv(
        "DATA_DIR",
        os.path.join(
            woodgate_base_dir,
            "data"
        )
    )

    #: The `output_dir` attribute represents the path to a
    # directory on the host file system where the build output
    #: will be stored. This attribute is set via the
    #: `OUTPUT_DIR` environment variable. The output directory
    #: should be the parent directory to `build_dir`
    #: (`BUILD_DIR`). If the `OUTPUT_DIR` environment variable
    #: is not set, then the `output_dir` attribute is set to
    #: `$WOODGATE_BASE_DIR/output` by default. The program
    #: will attempt to create `OUTPUT_DIR` if it does not
    #: already exist.
    output_dir: str = os.getenv(
        "OUTPUT_DIR",
        os.path.join(
            woodgate_base_dir,
            "output"
        )
    )

    #: The `build_dir` attribute represents the path to a
    #: directory on the host file system where the versioned
    #: build output will be stored. Versioned build output
    #: simply means that individual builds are namespaced by
    #: version within the `$OUTPUT_DIR`. This attribute is
    #: set via the `BUILD_DIR` environment variable.
    #: For example, if the `output_dir` attribute takes the
    #: value `$OUTPUT_DIR` then the `build_dir` attribute
    #: should look like `$OUTPUT_DIR/%Y%m%d-%H%M%s` for a
    #: string formatted date `%Y%m%d-%H%M%s`.
    #: The program will attempt to create `BUILD_DIR` if it
    #: does not already exist.
    build_dir: str = os.getenv(
        "BUILD_DIR",
        os.path.join(
            output_dir,
            build_version
        )
    )

    #: The `model_build_dir` attribute represents the path
    #: to a directory on the host file system where the
    #: directory learning model will be stored. The model
    #: build directory should be a child directory of the
    #: build directory in which the program will store the
    #: tuned model after training. This attribute is set via
    #: the `MODEL_BUILD_DIR` environment variable. If the
    #: `MODEL_BUILD_DIR` environment variable is not set,
    #: then the `model_build_dir` attribute will default to
    #: `$BUILD_DIR/$MODEL_NAME`. The program will attempt to
    #: create `MODEL_BUILD_DIR` if it does not already exist.
    model_build_dir: str = os.getenv(
        "MODEL_BUILD_DIR",
        os.path.join(
            build_dir,
            model_name
        )
    )

    #: The `model_data_dir` attribute represents the path
    #: to a directory on the host file system where the fine
    #: tuning data will be stored. The model data directory
    #: should be a child directory of the data directory in
    #: which the program will store the tuned model after
    #: training. This attribute is set via the
    #: `MODEL_DATA_DIR` environment variable. If the
    #: `MODEL_DATA_DIR` environment variable is not set, then
    #: the `model_data_dir` attribute will default to
    #: `$DATA_DIR/$MODEL_NAME`. The program will attempt to
    #: create `MODEL_DATA_DIR` if it does not already exist.
    model_data_dir: str = os.getenv(
        "MODEL_DATA_DIR",
        os.path.join(
            data_dir,
            model_name
        )
    )

    #: The `log_dir` attribute represents the path to a
    #: directory on the host file system where the
    #: log data will be stored. The log data directory should
    #: be a child directory of the build output directory
    #: in which the program will store the data retrieved for
    #: logging the build process. This attribute is set via
    #: the `LOG_DIR` environment variable. If the `LOG_DIR`
    #: environment variable is not set, then the `log_dir`
    #: attribute will default to `$OUTPUT_DIR/log`. The
    #: program will attempt to create `LOG_DIR` if it does not
    #: already exist.
    log_dir: str = os.getenv(
        "LOG_DIR",
        os.path.join(
            output_dir,
            "log",
            build_version
        )
    )

    #: The `log_file` attribute represents the base name of
    #: the `log_path` attribute. The log file should therefore
    #: reside in the `log_dir` by definition. This file should
    #: have a CSV file (having a `.csv` file extension). This
    #: attribute is set via the `REGRESSION_FILE` environment
    #: variable. If the `LOG_FILE` environment variable is not
    #: set, then the `log_file` will default to
    #: `$BUILD_VERSION.log`.
    log_file: str = os.getenv(
        "LOG_FILE",
        f"{build_version}.log"
    )

    #: The `training_dir` attribute represents the path to a
    #: directory on the host file system where the
    #: training data will be stored. The training data
    #: directory should be a child directory of the model data
    #: directory in which the program will store the data
    #: retrieved for training the learning model. This
    #: attribute is set via the `TRAINING_DIR` environment
    #: variable. If the `TRAINING_DIR` environment variable
    #: is not set, then the `training_dir` attribute will
    #: default to `$MODEL_DATA_DIR/train`. The program will
    #: attempt to create `TRAINING_DIR` if it does not
    #: already exist.
    training_dir: str = os.getenv(
        "TRAINING_DIR",
        os.path.join(
            model_data_dir,
            "train"
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
    training_file: str = os.getenv(
        "TRAINING_FILE",
        "train.csv"
    )

    #: The `testing_dir` attribute represents the path to a
    #: directory on the host file system where the
    #: testing data will be stored. The testing data directory
    #: should be a child directory of the model data
    #: directory in which the program will store the data
    #: retrieved for testing the learning model. This
    #: attribute is set via the `TESTING_DIR` environment
    #: variable. If the `TESTING_DIR` environment variable is
    #: not set, then the `testing_dir` attribute will default
    #: to `$MODEL_DATA_DIR/test`. The program will attempt to
    #: create `TESTING_DIR` if it does not already exist.
    testing_dir: str = os.getenv(
        "TESTING_DIR",
        os.path.join(
            model_data_dir,
            "test"
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
    testing_file: str = os.getenv(
        "TESTING_FILE",
        "test.csv"
    )

    #: The `evaluation_dir` attribute represents the path to
    #: a directory on the host file system where the
    #: evaluation data will be stored. The evaluation data
    #: directory should be a child directory of the model data
    #: directory in which the program will store the data
    #: retrieved for validating the learning model. This
    #: attribute is set via the `EVALUATION_DIR` environment
    #: variable. If the `EVALUATION_DIR` environment variable
    #: is not set, then the `evaluation_dir` attribute will
    #: default to `$MODEL_DATA_DIR/evaluate`. The program will
    #: attempt to create `EVALUATION_DIR` if it does not
    #: already exist.
    evaluation_dir: str = os.getenv(
        "EVALUATION_DIR",
        os.path.join(
            model_data_dir,
            "evaluate"
        )
    )

    #: The `evaluation_file` attribute represents the base
    #: name of the `evaluation_path` attribute. The evaluation
    #: file should therefore reside in the `evaluation_dir`
    #: by definition. This file should have a CSV file (having
    #: a `.csv` file extension). This attribute is set via
    #: the `EVALUATION_FILE` environment variable. The
    #: evaluation file should contain at least two (2)
    #: columns; one (1) for text having title
    #: `text_column_title` (set via `TEXT_COLUMN_TITLE`
    #: environment variable) and one (1) for label having
    #: title `label_column_title` (set via
    #: `LABEL_COLUMN_TITLE` environment variable).
    evaluation_file: str = os.getenv(
        "EVALUATION_FILE",
        "evaluate.csv"
    )

    #: The `regression_dir` attribute represents the path to
    #: a directory on the host file system where the
    #: regression data will be stored. The regression data
    #: directory should be a child directory of the model data
    #: directory in which the program will store the data
    #: retrieved for validating the learning model. This
    #: attribute is set via the `REGRESSION_DIR` environment
    #: variable. If the `REGRESSION_DIR` environment variable
    #: is not set, then the `regression_dir` attribute will
    #: default to `$MODEL_DATA_DIR/regress`. The program will
    #: attempt to create `REGRESSION_DIR` if it does not
    #: already exist.
    regression_dir: str = os.getenv(
        "REGRESSION_DIR",
        os.path.join(
            model_data_dir,
            "regress"
        )
    )

    #: The `regression_file` attribute represents the base
    #: name of the `regression_path` attribute. The regression
    #: file should therefore reside in the `regression_dir`
    #: by definition. This file should have a CSV file (having
    #: a `.csv` file extension). This attribute is set via
    #: the `REGRESSION_FILE` environment variable. The
    #: regression file should contain at least two (2) columns
    #: ; one (1) for text having title `text_column_title`
    #: (set via `TEXT_COLUMN_TITLE` environment variable)
    #: and one (1) for label having title `label_column_title`
    #: (set via `LABEL_COLUMN_TITLE` environment variable).
    regression_file: str = os.getenv(
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
    datasets_summary_dir: str = os.getenv(
        "DATASETS_SUMMARY_DIR",
        os.path.join(
            build_dir,
            "datasets_summary"
        )
    )

    #: The `build_summary_dir` attribute represents a
    #: directory on the host's file system. This is where the
    #: summary files generated by the
    #: `woodgate.build.build_summary.create_*` methods
    #: are stored. The program will attempt to
    #: create `BUILD_SUMMARY_DIR` if it does not already
    #: exist.
    build_summary_dir: str = os.getenv(
        "BUILD_SUMMARY_DIR",
        os.path.join(
            build_dir,
            "build_summary"
        )
    )

    #: The `evaluation_summary_dir` attribute represents a
    #: directory on the host's file system. This is where the
    #: summary files generated by the
    #: `woodgate.model.evaluation_summary.create_*` methods
    #: are stored. The program will attempt to
    #: create `EVALUATION_SUMMARY_DIR` if it does not already
    #: exist.
    evaluation_summary_dir: str = os.getenv(
        "EVALUATION_SUMMARY_DIR",
        os.path.join(
            build_dir,
            "evaluation_summary"
        )
    )

    #: The `bert_dir` attribute represents a directory on the
    #: host file system containing the BERT transfer model and
    #: associated files. This attribute is set via the `BERT_DIR`
    #: environment variable. If the `BERT_DIR` environment
    #: variable is not set, then the `bert_dir` attribute
    #: defaults to `$WOODGATE_BASE_DIR/bert`. The program will
    #: attempt to create `BERT_DIR` if it does not already exist.
    bert_dir: str = os.getenv(
        "BERT_DIR",
        os.path.join(
            woodgate_base_dir,
            "bert"
        )
    )

    #: The `bert_config_file` attribute represents the name of
    #: the BERT configuration JSON file. This attribute is set
    #: via the `BERT_CONFIG_FILE` environment variable. If the
    #: `BERT_CONFIG_FILE` environment variable is not set, then
    #: the `bert_config_file` defaults to `bert_config.json`.
    bert_config_file: str = os.getenv(
        "BERT_CONFIG_FILE",
        "bert_config.json"
    )

    #: The `bert_model_file` attribute represents the name of the
    #: BERT model (`.ckpt` file extension). This attribute is set
    #: via the `BERT_MODEL_FILE` environment variable. If the
    #: `BERT_MODEL_FILE` environment variable is not set, then
    #: the `bert_model_file` defaults to `bert_model.ckpt`.
    bert_model_file: str = os.getenv(
        "BERT_MODEL_FILE",
        "bert_model.ckpt"
    )

    #: The `bert_vocab_file` attribute represents the name of the
    #: BERT model's vocabulary file (`.txt` file extension).
    #: This attribute is set via the `BERT_VOCAB_FILE`
    #: environment variable. If the `BERT_VOCAB_FILE` environment
    #: variable is not set, then the `bert_vocab_file` defaults
    #: to `vocab.txt`.
    bert_vocab_file: str = os.getenv(
        "BERT_VOCAB_FILE",
        "vocab.txt"
    )

    #: The `create_tensorboard_logs` attribute represents a
    #: signal variable that is used to decide whether
    #: tensorboard logs should be generated along with build
    #: logs. This attribute is set via the
    #: `CREATE_TENSORBOARD_LOGS` environment
    #: variable. If the `CREATE_TENSORBOARD_LOGS` environment
    #: variable is not set, then the `create_tensorboard_logs`
    #: attribute is set to `1` by default signaling the
    #: program to generate tensorboard logs. All values
    #: except `CREATE_TENSORBOARD_LOGS=0` signal the program
    #: to generate tensorboard logs.
    create_tensorboard_logs = int(
        os.getenv("CREATE_TENSORBOARD_LOGS", "1")
    )

    #: The `validation_split` attribute represents a decimal
    #: number between 0 and 1. This attribute is set via the
    #: `VALIDATION_SPLIT` environment variable.
    #: Validation split indicates the proportional split of your
    #: training set by the value of the variable.
    #: For example, a value of `VALIDATION_SPLIT=0.2`
    #: would signal the program to reserve 20% of the
    #: training set for validation testing
    #: completed after each training epoch. If the
    #: `VALIDATION_SPLIT` environment variable is not set,
    #: then the `validation_split` attribute will default to
    #: `0.1`.
    validation_split: float = float(
        os.getenv("VALIDATION_SPLIT", "0.1")
    )

    #: The `batch_size` attribute represents an integer number
    #: between 8 and 512 inclusive. This value indicates the
    #: number of training examples utilized in one iteration.
    #: The batch size is a characteristic of gradient descent
    #: training algorithms. If the `BATCH_SIZE` environment
    #: variable is not set, then the `batch_size` attribute
    #: will default to `16`.
    batch_size: int = int(
        os.getenv("BATCH_SIZE", "16")
    )

    #: The `epochs` attribute represents an integer between
    #: 1-1000 inclusive.
    #: This attribute is set via the `EPOCHS` environment
    #: variable. This value indicates the number of
    #: times the training algorithm will iterate over the
    #: training dataset before completing. If the `EPOCHS`
    #: environment variable is unset, then the `epochs`
    #: attribute will default to `5`.
    epochs: int = int(
        os.getenv("EPOCHS", "1")
    )

    #: The `clf_out_dropout_rate` attribute represents one
    #: of two (1 / 2) dropout rates which may be customized.
    #: Typically this value should be around `0.5`. This
    #: attribute is set via the `CLF_OUT_DROPOUT_RATE`
    #: environment variable. If the `CLF_OUT_DROPOUT_RATE`
    #: environment variable is not set, then the
    #: `clf_out_dropout_rate` defaults to `0.5`.
    clf_out_dropout_rate: float = float(
        os.getenv(
            "CLF_OUT_DROPOUT_RATE",
            "0.5"
        )
    )

    #: The `clf_out_activation` attribute represents one
    #: of two (1 / 2) activation functions which may be
    #: customized. This attribute is set via the
    #: `CLF_OUT_ACTIVATION` environment variable. If the
    #: `CLF_OUT_ACTIVATION` environment variable is not set,
    #: then the `clf_out_activation` defaults to `tanh`.
    clf_out_activation: str = os.getenv(
        "CLF_OUT_ACTIVATION",
        "tanh"
    )

    #: The `logits_dropout_rate` attribute represents two
    #: of two (2 / 2) dropout rates which may be customized.
    #: Typically this value should be around `0.5`. This
    #: attribute is set via the `LOGITS_DROPOUT_RATE`
    #: environment variable. If the `LOGITS_DROPOUT_RATE`
    #: environment variable is not set, then the
    #: `logits_dropout_rate` defaults to `0.5`.
    logits_dropout_rate: float = float(
        os.getenv(
            "LOGITS_DROPOUT_RATE",
            "0.5"
        )
    )

    #: The `logits_activation` attribute represents two
    #: of two (2 / 2) activation functions which may be
    #: customized. This attribute is set via the
    #: `LOGITS_ACTIVATION` environment variable. If the
    #: `LOGITS_ACTIVATION` environment variable is not set,
    #: then the `logits_activation` defaults to `tanh`.
    logits_activation: str = os.getenv(
        "LOGITS_ACTIVATION",
        "softmax"
    )

    #: The `optimizer_name` attribute represents the optimization
    #: algorithm employed by the compilation process. This
    #: attribute is set via the `OPTIMIZER_NAME` environment
    #: variable. If the `OPTIMIZER_NAME` environment variable is
    #: not set, then the `optimizer_name` attribute defaults to
    #: `Adam` representing the Adam optimizer.
    #: The goal is to support all optimizers found in
    #: `tf.keras.optimizers`. The name reference the default
    #: name value. Thus, the best way to determine the value of
    #: `OPTIMIZER_NAME` for a given optimizer is to refer to the
    #: source module `tf.keras.optimizers` and find the class
    #: definition of the desired optimizer and use the default
    #: value of the `name` argument. The default values are
    #: sufficient for most cases. The attribute is case
    #: insensitive. USE WITH CAUTION!
    optimizer_name: str = os.getenv(
        "OPTIMIZER_NAME",
        "Adam"
    )

    #: The `optimizer_learning_rate` attribute represents a
    #: floating point value which represents the step size taken
    #: by the optimization algorithm toward a minimum loss.
    #: The `optimizer_learning_rate` attribute is set via the
    #: `OPTIMIZER_LEARNING_RATE` environment variable
    #: This value should be in the interval (0-1], otherwise a
    #: ValueError will be thrown at run time.
    optimizer_learning_rate: float = float(
        os.getenv("OPTIMIZER_LEARNING_RATE", "1e-5")
    )

    #: The `optimizer_args` attribute represents the additional
    #: arguments that may be passed to the selected optimizer.
    #: It is up to the user to know the order of additional
    #: optimizer argument, and which optimizer requires which
    #: values. If additional arguments are not required, then
    #: this attribute will not effect the instantiation of the
    #: requested optimizer. In general these should be a comma
    #: separated list of floating point (decimal) number enclosed
    #: in square brackets.
    optimizer_args: List[float] = ast.literal_eval(
        os.getenv(
            "OPTIMIZER_ARGS",
            "[]"
        )
    )

    #: The `loss_name` attribute represents the optimization
    #: algorithm employed by the compilation process. This
    #: attribute is set via the `LOSS_NAME` environment
    #: variable. If the `LOSS_NAME` environment variable is
    #: not set, then the `loss_name` attribute defaults to
    #: `Sparse_Categorical_Crossentropy` representing the
    #: Sparse Categorical Crossentropy loss algorithm.
    #: The goal is to support all losses found in
    #: `tf.keras.losses`. The name reference the default
    #: name value. Thus, the best way to determine the value of
    #: `LOSS_NAME` for a given loss is to refer to the
    #: source module `tf.keras.losses` and find the class
    #: definition of the desired loss and use the default
    #: value of the `name` argument. The default values are
    #: sufficient for most cases. The attribute is case
    #: insensitive. USE WITH CAUTION!
    loss_name: str = os.getenv(
        "LOSS_NAME",
        "Sparse_Categorical_Crossentropy"
    )

    #: The `loss_args` attribute represents the additional
    #: arguments that may be passed to the selected optimizer.
    #: It is up to the user to know the order of additional
    #: optimizer argument, and which optimizer requires which
    #: values. If additional arguments are not required, then
    #: this attribute will not effect the instantiation of the
    #: requested optimizer. In general these should be a comma
    #: separated list of floating point (decimal) number
    #: enclosed in square brackets.
    loss_args: List[Any] = ast.literal_eval(
        os.getenv(
            "LOSS_ARGS",
            "[True]"
        )
    )

    #: The `optimizer_metrics` attribute represents the
    #: metrics recorded during evaluation of the model.
    #: This attribute is
    #: In general these should be a comma separated list of
    #: strings (text) enclosed in square brackets; for each loss
    #: function there (usually) exists a corresponding metric.
    #: Thus, the best way to determine the value of
    #: `OPTIMIZER_METRICS` for a given loss is to
    #: refer to the source module `tf.keras.losses` and find the
    #: class definition of the desired metric and use the default
    #: value of the `name` argument. The default values are
    #: sufficient for most cases. The list items are case
    #: insensitive. USE WITH CAUTION!
    optimizer_metrics: List[str] = ast.literal_eval(
        os.getenv(
            "OPTIMIZER_METRICS",
            "['Sparse_Categorical_Crossentropy']"
        )
    )

    @classmethod
    def get_log_path(cls) -> str:
        """The `get_log_path` method returns the full path on
        the host file system pointing to `log_file`.

        :return:
        :rtype:
        """
        return os.path.join(cls.log_dir, cls.log_file)

    @classmethod
    def get_training_path(cls) -> str:
        """The `get_training_path` method returns the full path
        on the host file system pointing to `training_file`.

        :return: Path to `cls.training_file`
        :rtype: str
        """

        return os.path.join(cls.training_dir, cls.training_file)

    @classmethod
    def get_testing_path(cls) -> str:
        """The `get_testing_path` method returns the full path
        on the host file system pointing to `testing_file`.

        :return: Path to `cls.testing_file`
        :rtype: str
        """

        return os.path.join(cls.testing_dir, cls.testing_file)

    @classmethod
    def get_regression_path(cls) -> str:
        """The `get_regression_path` method returns the full path
        on the host file system pointing to `regression_file`.

        :return: Path to `cls.regression_file`
        :rtype: str
        """

        return os.path.join(
            cls.regression_dir, cls.regression_file)

    @classmethod
    def get_evaluation_path(cls) -> str:
        """The `get_evaluation_path` method returns the full path
        on the host file system pointing to `evaluation_file`.

        :return: Path to `cls.evaluation_file`
        :rtype: str
        """

        return os.path.join(
            cls.evaluation_dir, cls.evaluation_file)

    @classmethod
    def get_bert_config_path(cls) -> str:
        """The `get_bert_config_path` method returns the full
        path on the host file system pointing to
        `bert_config_file`.

        :return: Path to `cls.bert_config_file`
        :rtype: str
        """
        return os.path.join(cls.bert_dir, cls.bert_config_file)

    @classmethod
    def get_bert_model_path(cls) -> str:
        """The `get_bert_model_path` method returns the full
        path on the host file system pointing to
        `bert_config_file`.

        :return: Path to `cls.bert_config_file`
        :rtype: str
        """

        return os.path.join(cls.bert_dir, cls.bert_model_file)

    @classmethod
    def get_bert_vocab_path(cls):
        """The `get_bert_vocab_path` method returns the full
        path on the host file system pointing to
        `bert_vocab_file`.

        :return: Path to `cls.bert_vocab_file`
        :rtype: str
        """
        return os.path.join(cls.bert_dir, cls.bert_vocab_file)
