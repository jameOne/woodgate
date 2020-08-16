"""
file_system_configuration.py - The file_system_configuration.py
module contains the FileSystemConfiguration class definition.
"""
import os
import datetime
import uuid


class FileSystemConfiguration:
    """
    FileSystemConfiguration - The FileSystemConfiguration class
    encapsulates logic related to configuring the model builder.
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
        datetime.datetime.now().strftime("%Y_%m_%d-%H:%M:%S")
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
    os.makedirs(woodgate_base_dir, exist_ok=True)

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
    os.makedirs(data_dir, exist_ok=True)

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
    os.makedirs(output_dir, exist_ok=True)

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
    os.makedirs(build_dir, exist_ok=True)

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
    os.makedirs(model_build_dir, exist_ok=True)

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
    os.makedirs(output_dir, exist_ok=True)

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
    os.makedirs(log_dir, exist_ok=True)

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

    #: The `log_path` attribute represents the full path on
    #: the host file system pointing to `log_file`. This
    #: attribute is set via the `LOG_PATH` environment
    #: variable. If `LOG_PATH` is set, then it will render
    #: values set by `log_dir` and `log_file` inconsequential.
    #: If `LOG_PATH` is not set, then the `log_path` will
    #: default to `$LOG_DIR/$LOG_FILE`.
    log_path: str = os.path.join(
        log_dir,
        log_file
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
    os.makedirs(training_dir, exist_ok=True)

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

    #: The `training_path` attribute represents the full path
    #: on the host file system pointing to `training_file`.
    #: This attribute is set via the `TRAINING_PATH`
    #: environment variable. If `TRAINING_PATH` is set, then
    #: it will make values set by `training_dir` and
    #: `training_file` inconsequential. If `TRAINING_PATH` is
    #: not set, then the `training_path` will default to
    #: `$TRAINING_DIR/$TRAINING_FILE`.
    training_path: str = os.getenv(
        "TRAINING_PATH",
        os.path.join(
            training_dir,
            training_file
        )
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
    os.makedirs(testing_dir, exist_ok=True)

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

    #: The `testing_path` attribute represents the full path
    #: on the host file system pointing to `testing_file`.
    #: This attribute is set via the `TESTING_PATH`
    #: environment variable. If `TESTING_PATH` is set, then it
    #: will make values set by `testing_dir` and
    #: `testing_file` inconsequential. If `TESTING_PATH` is
    #: not set, then the `testing_path` will default to
    #: `$TESTING_DIR/$TESTING_FILE`.
    testing_path: str = os.path.join(
        testing_dir,
        testing_file
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
    os.makedirs(evaluation_dir, exist_ok=True)

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

    #: The `evaluation_path` attribute represents the full
    #: path on the host file system pointing to
    #: `evaluation_file`. This attribute is set via the
    #: `EVALUATION_PATH` environment variable. If
    #: `EVALUATION_PATH` is set, then it will make values
    #: set by `evaluation_dir` and `evaluation_file`
    #: inconsequential. If `EVALUATION_PATH` is not set,
    #: then the `evaluation_path` will default to
    #: `$EVALUATION_DIR/$EVALUATION_FILE`.
    evaluation_path: str = os.path.join(
        evaluation_dir,
        evaluation_file
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
    os.makedirs(regression_dir, exist_ok=True)

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

    #: The `regression_path` attribute represents the full
    #: path on the host file system pointing to
    #: `regression_file`. This attribute is set via the
    #: `REGRESSION_PATH` environment variable. If
    #: `REGRESSION_PATH` is set, then it will make values
    #: set by `regression_dir` and `regression_file`
    #: inconsequential. If `REGRESSION_PATH` is not set,
    #: then the `regression_path` will default to
    #: `$REGRESSION_DIR/$REGRESSION_FILE`.
    regression_path: str = os.path.join(
        regression_dir,
        regression_file
    )

    #: The `datasets_summary_dir` attribute represents a
    #: directory on the host's file system. This is where the
    #: summary files generated by the `create_*` methods
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
    os.makedirs(datasets_summary_dir, exist_ok=True)

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
    os.makedirs(bert_dir, exist_ok=True)

    #: The `bert_config_file` attribute represents the name of
    #: the BERT configuration JSON file. This attribute is set
    #: via the `BERT_CONFIG_FILE` environment variable. If the
    #: `BERT_CONFIG_FILE` environment variable is not set, then
    #: the `bert_config_file` defaults to `bert_config.json`.
    bert_config_file: str = os.getenv(
        "BERT_CONFIG_FILE",
        "bert_config.json"
    )

    #: The `bert_config_path` attribute represents the full
    #: path on the host file system pointing to
    #: `bert_config_file`. This attribute is set via the
    #: `BERT_CONFIG_PATH` environment variable. If
    #: `BERT_CONFIG_PATH` is set, then it will make values
    #: set by `bert_config_dir` and `bert_config_file`
    #: inconsequential. If `BERT_CONFIG_PATH` is not set,
    #: then the `bert_config_path` will default to
    #: `$BERT_CONFIG_DIR/$BERT_CONFIG_FILE`.
    bert_config_path: str = os.getenv(
        "BERT_CONFIG_PATH",
        os.path.join(
            bert_dir,
            bert_config_file
        )
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

    #: The `bert_model_path` attribute represents the full
    #: path on the host file system pointing to
    #: `bert_config_file`. This attribute is set via the
    #: `BERT_MODEL_PATH` environment variable. If
    #: `BERT_MODEL_PATH` is set, then it will make values
    #: set by `bert_model_dir` and `bert_model_file`
    #: inconsequential. If `BERT_MODEL_PATH` is not set,
    #: then the `bert_model_path` will default to
    #: `$BERT_MODEL_DIR/$BERT_MODEL_FILE`.
    bert_model_path: str = os.getenv(
        "BERT_MODEL_PATH",
        os.path.join(
            bert_dir,
            bert_model_file
        )
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

    #: The `bert_vocab_path` attribute represents the full
    #: path on the host file system pointing to
    #: `bert_vocab_file`. This attribute is set via the
    #: `BERT_VOCAB_PATH` environment variable. If
    #: `BERT_VOCAB_PATH` is set, then it will make values
    #: set by `bert_vocab_dir` and `bert_vocab_file`
    #: inconsequential. If `BERT_VOCAB_PATH` is not set,
    #: then the `bert_vocab_path` will default to
    #: `$BERT_VOCAB_DIR/$BERT_VOCAB_FILE`.
    bert_vocab_path: str = os.getenv(
        "BERT_VOCAB_PATH",
        os.path.join(
            bert_dir,
            bert_vocab_file
        )
    )
