"""
build_configuration.py - The build_configuration.py module contains the BuildConfiguration class definition.
"""
import os
import datetime
import uuid


class BuildConfiguration:
    """
    BuildConfiguration - The BuildConfiguration class encapsulates logic related to configuring the model builder.
    """

    def __init__(self):
        #: The `model_name` attribute represents the name given to the machine learning model.
        #: This attribute is set via the `MODEL_NAME` environment variable. If the `MODEL_NAME` environment variable is
        #: not set, the `model_name` attribute is set to a random (v4) UUID by default.
        self.model_name: str = os.getenv("MODEL_NAME", str(uuid.uuid4()))

        #: The `build_version` attribute represents the specific version of the model build.
        #: This attribute is set via the `BUILD_VERSION` environment variable. If the `BUILD_VERSION` environment
        #: variable is not set, the `build_version` attribute is set to a string formatted time ("%Y%m%d-%H%M%s")
        #: by default.
        self.build_version: str = os.getenv("BUILD_VERSION", datetime.datetime.now().strftime("%Y%m%d-%H%M%s"))

        #: The create_dataset_visuals attribute represents a signal variable that is used
        #: to decide whether visuals should be generated along with a text summary of the training, testing, validation,
        #: evaluation, and regression datasets. This attribute is set via the `CREATE_DATASET_VISUALS` environment
        #: variable. If the `CREATE_DATASET_VISUALS` environment variable is not set, then
        #: the create_dataset_visuals attribute is set to `1` by default signaling the program to generate visuals.
        #: All values except `CREATE_DATASET_VISUALS=0` signal the program to generate dataset visuals.
        self.create_dataset_visuals: int = int(os.getenv("CREATE_DATASET_VISUALS", "1"))

        #: The `create_build_visuals` attribute represents a signal variable that is used
        #: to decide whether visuals should be generated along with a text summary of the build history. This attribute
        #: is set via the `CREATE_BUILD_VISUALS` environment variable. If the
        #: `CREATE_BUILD_VISUALS` environment variable is not set, then the `create_build_visuals` attribute is
        #: set to `1` by default signaling the program to generate visuals. All values except `CREATE_DATASET_VISUALS=0`
        #: signal the program to generate build visuals.
        self.create_build_visuals: int = int(os.getenv("CREATE_BUILD_VISUALS", "1"))

        #: The `woodgate_base_dir` attribute represents the path to a directory on the host file system from which
        #: many of the other file paths are constructed. This directory is also where files generated or downloaded by
        #: the process are stored. This attribute is set via the `WOODGATE_BASE_DIR` environment variable. If the
        #: `WOODGATE_BASE_DIR` environment variable is not set, then the `woodgate_base_dir` attribute is set to
        #: `~/woodgate`. The program will attempt to create `WOODGATE_BASE_DIR` if it does not already exist.
        self.woodgate_base_dir: str = os.getenv("WOODGATE_BASE_DIR", "~/woodgate")
        os.makedirs(self.woodgate_base_dir, exist_ok=True)

        #: The `bert_dir` attribute represents the path to a directory on the host file system containing the
        #: BERT model. This attribute is set via the `BERT_DIR` environment variable.
        #: For example, consider the following script:
        #: ```# Download BERT model
        #:  wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip
        #:
        #:  mkdir ~/models
        #:  mkdir ~/models/bert
        #:
        #: # Unzip the file
        #:  unzip uncased_L-12_H-768_A-12.zip -d ~/models/bert```
        #:
        #: ~/models/bert would be the proper bert_dir environment variable would be `BERT_DIR=~/models/bert`.
        #: If the `BERT_DIR` environment variable is not set, then the `bert_dir` attribute defaults to:
        #: `$WOODGATE_BASE_DIR/bert`. The program will attempt to create `BERT_DIR` if it does not already
        #: exist.
        self.bert_dir: str = os.getenv("BERT_DIR", os.path.join(self.woodgate_base_dir, "bert"))
        os.makedirs(self.bert_dir, exist_ok=True)

        #: The `data_dir` attribute represents the path to a directory on the host file system where data files
        #: are stored. This attribute is set via the `DATA_DIR` environment variable. The data directory should be the
        #: parent directory of `training_dir` (`TRAINING_DIR`), `testing_dir` (`TESTING_DIR`), `evaluation_dir`
        #: (`EVALUATION_DIR`), `validation_dir` (`VALIDATION_DIR`) and `regression_dir` (`REGRESSION_DIR`).
        #: If the `DATA_DIR` is not set, then the `data_dir` attribute is set to `$WOODGATE_BASE_DIR/data` by default.
        #: The program will attempt to create `DATA_DIR` if it does not already exist.
        self.data_dir: str = os.getenv("DATA_DIR", os.path.join(self.woodgate_base_dir, "data"))
        os.makedirs(self.data_dir, exist_ok=True)

        #: The `output_dir` attribute represents the path to a directory on the host file system where the build output
        #: will be stored. This attribute is set via the `OUTPUT_DIR` environment variable. The output directory should
        #: be the parent directory to `build_dir` (`BUILD_DIR`). If the `OUTPUT_DIR` environment variable is not set,
        #: then the `output_dir` attribute is set to `$WOODGATE_BASE_DIR/output` by default. The program will attempt to
        #: create `OUTPUT_DIR` if it does not already exist.
        self.output_dir: str = os.getenv("OUTPUT_DIR", os.path.join(self.woodgate_base_dir, "output"))
        os.makedirs(self.output_dir, exist_ok=True)

        #: The `build_dir` attribute represents the path to a directory on the host file system where the versioned
        #: build output will be stored. Versioned build output simply means that individual builds are namespaced by
        #: version within the `$OUTPUT_DIR`. This attribute is set via the `BUILD_DIR` environment variable.
        #: For example, if the `output_dir` attribute takes the value `$OUTPUT_DIR` then the `build_dir` attribute
        #: should look like `BUILD_DIR=$OUTPUT_DIR/%Y%m%d-%H%M%s` for a string formatted date `%Y%m%d-%H%M%s`.
        #: The program will attempt to create `BUILD_DIR` if it does not already exist.
        self.build_dir: str = os.getenv("BUILD_DIR", os.path.join(self.output_dir, self.build_version))
        os.makedirs(self.build_dir, exist_ok=True)

        #: The `model_build_dir` attribute represents the path to a directory on the host file system where the learning
        #: model will be stored. The model build directory should be a child directory of the build directory in which
        #: the program will store the tuned model after training. This attribute is set via the `MODEL_BUILD_DIR`
        #: environment variable. If the `MODEL_BUILD_DIR` environment variable is not set, then the `model_build_dir`
        #: attribute will default to `$BUILD_DIR/$MODEL_NAME`. The program will attempt to create `MODEL_BUILD_DIR` if
        #: it does not already exist.
        self.model_build_dir: str = os.getenv("MODEL_BUILD_DIR", os.path.join(self.build_dir, self.model_name))
        os.makedirs(self.output_dir, exist_ok=True)

        #: The `model_data_dir` attribute represents the path to a directory on the host file system where the fine
        #: tuning data will be stored. The model data directory should be a child directory of the data directory in
        #: which the program will store the tuned model after training. This attribute is set via the `MODEL_DATA_DIR`
        #: environment variable. If the `MODEL_DATA_DIR` environment variable is not set, then the `model_data_dir`
        #: attribute will default to `$DATA_DIR/$MODEL_NAME`. The program will attempt to create `MODEL_DATA_DIR` if
        #: it does not already exist.
        self.model_data_dir: str = os.getenv("MODEL_DATA_DIR", os.path.join(self.data_dir, self.model_name))
        os.makedirs(self.output_dir, exist_ok=True)

        #: The `training_dir` attribute represents the path to a directory on the host file system where the
        #: training data will be stored. The training data directory should be a child directory of the model data
        #: directory in which the program will store the data retrieved for training the learning model. This attribute
        #: is set via the `TRAINING_DIR` environment variable. If the `TRAINING_DIR` environment variable is not
        #: set, then the `training_dir` attribute will default to `$MODEL_DATA_DIR/train`. The program will attempt to
        #: create `TRAINING_DIR` if it does not already exist.
        self.training_dir: str = os.getenv("TRAINING_DIR", os.path.join(self.model_data_dir, "train"))
        os.makedirs(self.training_dir, exist_ok=True)

        #: The `training_file` attribute represents the base name of the `training_path` attribute. The training file
        #: should therefore reside in the `training_dir` by definition. This file should have a CSV file (having a
        #: `.csv` file extension). This attribute is set via the `TRAINING_FILE` environment variable. The training
        #: file should contain at least two (2) columns; one (1) for text having title `text_column_title`
        #: (set via `TEXT_COLUMN_TITLE` environment variable) and one (1) for label having title `label_column_title`
        #: (set via `LABEL_COLUMN_TITLE` environment variable).
        self.training_file: str = os.getenv("TRAINING_FILE", "train.csv")

        #: The `training_path` attribute represents the full path on the host file system pointing to `training_file`.
        #: This attribute is set via the `TRAINING_PATH` environment variable. If `TRAINING_PATH` is set, then it will
        #: render values set by `training_dir` and `training_file` inconsequential. If `TRAINING_PATH` is not set, then
        #: the `training_path` will default to `$TRAINING_DIR/$TRAINING_FILE`.
        self.training_path: str = os.getenv("TRAINING_PATH", os.path.join(self.training_dir, self.training_file))

        #: The `testing_dir` attribute represents the path to a directory on the host file system where the
        #: testing data will be stored. The testing data directory should be a child directory of the model data
        #: directory in which the program will store the data retrieved for testing the learning model. This attribute
        #: is set via the `TESTING_DIR` environment variable. If the `TESTING_DIR` environment variable is not
        #: set, then the `testing_dir` attribute will default to `$MODEL_DATA_DIR/test`. The program will attempt to
        #: create `TESTING_DIR` if it does not already exist.
        self.testing_dir: str = os.getenv("TESTING_DIR", os.path.join(self.model_data_dir, "test"))
        os.makedirs(self.testing_dir, exist_ok=True)

        #: The `testing_file` attribute represents the base name of the `testing_path` attribute. The testing file
        #: should therefore reside in the `testing_dir` by definition. This file should have a CSV file (having a
        #: `.csv` file extension). This attribute is set via the `TESTING_FILE` environment variable. The testing
        #: file should contain at least two (2) columns; one (1) for text having title `text_column_title`
        #: (set via `TEXT_COLUMN_TITLE` environment variable) and one (1) for label having title `label_column_title`
        #: (set via `LABEL_COLUMN_TITLE` environment variable).
        self.testing_file: str = os.getenv("TESTING_FILE", "test.csv")

        #: The `testing_path` attribute represents the full path on the host file system pointing to `testing_file`.
        #: This attribute is set via the `TESTING_PATH` environment variable. If `TESTING_PATH` is set, then it will
        #: render values set by `testing_dir` and `testing_file` inconsequential. If `TESTING_PATH` is not set, then
        #: the `testing_path` will default to `$TESTING_DIR/$TESTING_FILE`.
        self.testing_path: str = os.path.join(self.testing_dir, self.testing_file)

        #: The `validation_dir` attribute represents the path to a directory on the host file system where the
        #: validation data will be stored. The validation data directory should be a child directory of the model data
        #: directory in which the program will store the data retrieved for validating the learning model. This
        #: attribute is set via the `VALIDATION_DIR` environment variable. If the `VALIDATION_DIR` environment variable
        #: is not set, then the `validation_dir` attribute will default to `$MODEL_DATA_DIR/validate`. The program will
        #: attempt to create `VALIDATION_DIR` if it does not already exist.
        self.validation_dir: str = os.getenv("VALIDATION_DIR", os.path.join(self.data_dir, self.model_name, "validate"))
        os.makedirs(self.validation_dir, exist_ok=True)

        #: The `validation_file` attribute represents the base name of the `validation_path` attribute. The validation
        #: file should therefore reside in the `validation_dir` by definition. This file should have a CSV file (having
        #: a `.csv` file extension). This attribute is set via the `VALIDATION_FILE` environment variable. The
        #: validation file should contain at least two (2) columns; one (1) for text having title `text_column_title`
        #: (set via `TEXT_COLUMN_TITLE` environment variable) and one (1) for label having title `label_column_title`
        #: (set via `LABEL_COLUMN_TITLE` environment variable).
        self.validation_file: str = os.getenv("VALIDATION_FILE", "validate.csv")

        #: The `validation_path` attribute represents the full path on the host file system pointing to
        #: `validation_file`. This attribute is set via the `VALIDATION_PATH` environment variable. If
        #: `VALIDATION_PATH` is set, then it will render values set by `validation_dir` and `validation_file`
        #: inconsequential. If  `VALIDATION_PATH` is not set, then the `validation_path` will default to
        #: `$VALIDATION_DIR/$VALIDATION_FILE`.
        self.validation_path: str = os.path.join(self.validation_dir, self.validation_file)

        #: The `evaluation_dir` attribute represents the path to a directory on the host file system where the
        #: evaluation data will be stored. The evaluation data directory should be a child directory of the model data
        #: directory in which the program will store the data retrieved for validating the learning model. This
        #: attribute is set via the `EVALUATION_DIR` environment variable. If the `EVALUATION_DIR` environment variable
        #: is not set, then the `evaluation_dir` attribute will default to `$MODEL_DATA_DIR/evaluate`. The program will
        #: attempt to create `EVALUATION_DIR` if it does not already exist.
        self.evaluation_dir: str = os.getenv("EVALUATION_DIR", os.path.join(self.data_dir, self.model_name, "evaluate"))
        os.makedirs(self.evaluation_dir, exist_ok=True)

        #: The `evaluation_file` attribute represents the base name of the `evaluation_path` attribute. The evaluation
        #: file should therefore reside in the `evaluation_dir` by definition. This file should have a CSV file (having
        #: a `.csv` file extension). This attribute is set via the `EVALUATION_FILE` environment variable. The
        #: evaluation file should contain at least two (2) columns; one (1) for text having title `text_column_title`
        #: (set via `TEXT_COLUMN_TITLE` environment variable) and one (1) for label having title `label_column_title`
        #: (set via `LABEL_COLUMN_TITLE` environment variable).
        self.evaluation_file: str = os.getenv("EVALUATION_FILE", "evaluate.csv")

        #: The `evaluation_path` attribute represents the full path on the host file system pointing to
        #: `evaluation_file`. This attribute is set via the `EVALUATION_PATH` environment variable. If
        #: `EVALUATION_PATH` is set, then it will render values set by `evaluation_dir` and `evaluation_file`
        #: inconsequential. If  `EVALUATION_PATH` is not set, then the `evaluation_path` will default to
        #: `$EVALUATION_DIR/$EVALUATION_FILE`.
        self.evaluation_path: str = os.path.join(self.evaluation_dir, self.evaluation_file)

        #: The `regression_dir` attribute represents the path to a directory on the host file system where the
        #: regression data will be stored. The regression data directory should be a child directory of the model data
        #: directory in which the program will store the data retrieved for validating the learning model. This
        #: attribute is set via the `REGRESSION_DIR` environment variable. If the `REGRESSION_DIR` environment variable
        #: is not set, then the `regression_dir` attribute will default to `$MODEL_DATA_DIR/regress`. The program will
        #: attempt to create `REGRESSION_DIR` if it does not already exist.
        self.regression_dir: str = os.getenv("REGRESSION_DIR", os.path.join(self.data_dir, self.model_name, "regress"))
        os.makedirs(self.regression_dir, exist_ok=True)

        #: The `regression_file` attribute represents the base name of the `regression_path` attribute. The regression
        #: file should therefore reside in the `regression_dir` by definition. This file should have a CSV file (having
        #: a `.csv` file extension). This attribute is set via the `REGRESSION_FILE` environment variable. The
        #: regression file should contain at least two (2) columns; one (1) for text having title `text_column_title`
        #: (set via `TEXT_COLUMN_TITLE` environment variable) and one (1) for label having title `label_column_title`
        #: (set via `LABEL_COLUMN_TITLE` environment variable).
        self.regression_file: str = os.getenv("REGRESSION_FILE", "regress.csv")

        #: The `regression_path` attribute represents the full path on the host file system pointing to
        #: `regression_file`. This attribute is set via the `REGRESSION_PATH` environment variable. If
        #: `REGRESSION_PATH` is set, then it will render values set by `regression_dir` and `regression_file`
        #: inconsequential. If  `REGRESSION_PATH` is not set, then the `regression_path` will default to
        #: `$REGRESSION_DIR/$REGRESSION_FILE`.
        self.regression_path: str = os.path.join(self.regression_dir, self.regression_file)

        #: The `log_dir` attribute represents the path to a directory on the host file system where the
        #: log data will be stored. The log data directory should be a child directory of the build output
        #: directory in which the program will store the data retrieved for logging the build process. This
        #: attribute is set via the `LOG_DIR` environment variable. If the `LOG_DIR` environment
        #: variable is not set, then the `log_dir` attribute will default to `$OUTPUT_DIR/log`. The program
        #: will attempt to create `LOG_DIR` if it does not already exist.
        self.log_dir: str = os.getenv("LOG_DIR", os.path.join(self.output_dir, "log", self.build_version))
        os.makedirs(self.log_dir, exist_ok=True)

        #: The `log_file` attribute represents the base name of the `log_path` attribute. The log
        #: file should therefore reside in the `log_dir` by definition. This file should have a CSV file (having
        #: a `.csv` file extension). This attribute is set via the `REGRESSION_FILE` environment variable. If the
        #: `LOG_FILE` environment variable is not set, then the `log_file` will default to `$BUILD_VERSION.log`.
        self.log_file: str = os.getenv("LOG_FILE", f"{self.build_version}.log")

        #: The `log_path` attribute represents the full path on the host file system pointing to
        #: `log_file`. This attribute is set via the `LOG_PATH` environment variable. If
        #: `LOG_PATH` is set, then it will render values set by `log_dir` and `log_file`
        #: inconsequential. If  `LOG_PATH` is not set, then the `log_path` will default to
        #: `$LOG_DIR/$LOG_FILE`.
        self.log_path: str = os.path.join(self.log_dir, self.log_file)

        #: The `validation_split` attribute represents a decimal number between 0 and 1. This value indicates
        #: the proportional split of your training set by the value of the variable. For example, a value of
        #: `VALIDATION_SPLIT=0.2` would signal the program to reserve 20% of the training set for validation testing
        #: completed after each training epoch. If the `VALIDATION_SPLIT` environment variable is not set, then the
        #: `validation_split` attribute will default to `0.1`.
        self.validation_split: float = float(os.getenv("VALIDATION_SPLIT", "0.1"))
        if self.validation_split < 0 or self.validation_split > 1:
            raise ValueError(
                "check VALIDATION_SPLIT env var: " +
                "validation split must be a floating point value between 0-1 inclusive")

        #: The `batch_size` attribute represents an integer number between 8 and 512 inclusive. This value indicates the
        #: number of training examples utilized in one iteration. The batch size is a characteristic of gradient descent
        #: training algorithms. If the `BATCH_SIZE` environment variable is not set, then the `batch_size` attribute
        #: will default to `16`.
        self.batch_size: int = int(os.getenv("BATCH_SIZE", "16"))
        if self.batch_size < 8 or self.batch_size > 512:
            raise ValueError(
                "check BATCH_SIZE env var: " +
                "batch size must be an integer value between 8-512")

        #: The `epochs` attribute represents an integer between 1-1000 inclusive. This value indicates the number of
        #: times the training algorithm will iterate over the training dataset before completing. If the `EPOCHS`
        #: environment variable is unset, then the `epochs` attribute will default to `5`.
        self.epochs: int = int(os.getenv("EPOCHS", "5"))
        if self.epochs < 1 or self.epochs > 1000:
            raise ValueError(
                "check EPOCHS env var: " +
                "epochs size must be an integer value between 0-1000")
