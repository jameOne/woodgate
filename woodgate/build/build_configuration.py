"""
build_configuration.py - The build_configuration.py module contains the BuildConfiguration class definition.
"""
import os
import sys
import datetime
import uuid

from .build_logger import BuildLogger


def access_hint(env_var):
    """

    :param env_var:
    :type env_var:
    :return:
    :rtype:
    """
    return (f"make sure the {env_var} environment variable is a directory on the host"
            "file system and the current user has read/write access")


class BuildConfiguration:
    """
    BuildConfiguration - The BuildConfiguration class encapsulates logic related to configuring the model builder.
    """

    def __init__(self):
        #: The `model_name` attribute represents the name given to the machine learning model.
        #: This attribute is set via the `MODEL_NAME` environment variable. If the `MODEL_NAME` environment variable is
        #: not set, this attribute is set to a random (v4) UUID by default.
        self.model_name: str = os.getenv("MODEL_NAME", str(uuid.uuid4()))

        #: The `build_version` attribute represents the specific version of the model build.
        #: This attribute is set via the `BUILD_VERSION` environment variable. If the `BUILD_VERSION` environment
        #: variable is not set, this attribute is set to a string formatted time ("%Y%m%d-%H%M%s").
        self.build_version: str = os.getenv("BUILD_VERSION", datetime.datetime.now().strftime("%Y%m%d-%H%M%s"))

        #: The create_dataset_visuals attribute represents a signal variable that is used
        #: to decide whether visuals should be generated along with a text summary of the training, testing, validation,
        #: evaluation, and regression datasets. This attribute is set via the `CREATE_DATASET_VISUALS` environment
        #: variable. If the `CREATE_DATASET_VISUALS` environment variable is not set, then
        #: the create_dataset_visuals attribute is set to `1` signaling the program to generate visuals. All
        #: values except `CREATE_DATASET_VISUALS=0` signal the program to generate dataset visuals.
        try:
            self.create_dataset_visuals: int = int(os.getenv("CREATE_DATASET_VISUALS", "1"))
        except ValueError as err:
            BuildLogger.LOGGER.error(err)
            BuildLogger.LOGGER.info(
                "the CREATE_DATASET_VISUALS environment variable should be " +
                "set either `CREATE_DATASET_VISUALS=0` to signal no visuals" +
                ", unset or `CREATE_DATASET_VISUALS=<any non-zero integer>` to signal dataset visuals")
            sys.exit(1)

        #: The `create_build_visuals` attribute represents a signal variable that is used
        #: to decide whether visuals should be generated along with a text summary of the build history. If the
        #: `CREATE_BUILD_VISUALS` environment variable is not set, then the `create_build_visuals` attribute is
        #: set to `1` signaling the program to generate visuals. All values except `CREATE_DATASET_VISUALS=0` signal the
        #: program to generate build visuals.
        try:
            self.create_build_visuals: int = int(os.getenv("CREATE_BUILD_VISUALS", "1"))
        except ValueError as err:
            BuildLogger.LOGGER.error(err)
            BuildLogger.LOGGER.info(
                "make sure the CREATE_BUILD_VISUALS environment variable is " +
                "set either `CREATE_BUILD_VISUALS=0` to signal no visuals" +
                ", or, unset or `CREATE_BUILD_VISUALS=<any non-zero integer>` to signal" +
                " the creation of dataset visuals")
            sys.exit(1)

        #: The `woodgate_base_dir` attribute represents the path to a directory on the host file system from which
        #: many of the other file paths are constructed. This directory is also where files generated or downloaded by
        #: the process are stored. This attribute is set via the `WOODGATE_BASE_DIR` environment variable. If the
        #: `WOODGATE_BASE_DIR` environment variable is not set, then the woodgate_base_dir attribute is set to
        #: `~/woodgate`
        self.woodgate_base_dir: str = os.getenv("WOODGATE_BASE_DIR", "~/woodgate")
        try:
            os.makedirs(self.woodgate_base_dir, exist_ok=True)
        except OSError as err:
            BuildLogger.LOGGER.error(err)
            BuildLogger.LOGGER.info(access_hint("WOODGATE_BASE_DIR"))
            sys.exit(1)

        # TODO - Document.
        self.bert_dir: str = os.getenv("BERT_DIR", os.path.join(self.woodgate_base_dir, "bert"))
        try:
            os.makedirs(self.bert_dir, exist_ok=True)
        except OSError as err:
            BuildLogger.LOGGER.error(err)
            BuildLogger.LOGGER.info(access_hint("BERT_DIR"))
            sys.exit(1)

        # TODO - Document.
        self.data_dir: str = os.getenv("DATA_DIR", os.path.join(self.woodgate_base_dir, "data"))
        try:
            os.makedirs(self.data_dir, exist_ok=True)
        except OSError as err:
            BuildLogger.LOGGER.error(err)
            BuildLogger.LOGGER.info(access_hint("DATA_DIR"))
            sys.exit(1)

        # TODO - Document.
        self.output_dir: str = os.getenv("OUTPUT_DIR", os.path.join(self.woodgate_base_dir, "output"))
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except OSError as err:
            BuildLogger.LOGGER.error(err)
            BuildLogger.LOGGER.info(access_hint("OUTPUT_DIR"))
            sys.exit(1)

        # TODO - Document.
        self.build_dir: str = os.getenv("BUILD_DIR", os.path.join(self.output_dir, self.build_version))
        try:
            os.makedirs(self.build_dir, exist_ok=True)
        except OSError as err:
            BuildLogger.LOGGER.error(err)
            BuildLogger.LOGGER.info(access_hint("BUILD_DIR"))
            sys.exit(1)

        # TODO - Document.
        self.testing_dir: str = os.getenv("TESTING_DIR", os.path.join(self.data_dir, self.model_name, "test"))
        try:
            os.makedirs(self.testing_dir, exist_ok=True)
        except OSError as err:
            BuildLogger.LOGGER.error(err)
            BuildLogger.LOGGER.info(access_hint("TESTING_DIR"))
            sys.exit(1)

        # TODO - Document.
        self.testing_file: str = os.getenv("TESTING_FILE", "test.csv")
        # TODO - Document.
        self.testing_path: str = os.path.join(self.testing_dir, self.testing_file)

        # TODO - Document.
        self.training_dir: str = os.getenv("TRAINING_DIR", os.path.join(self.data_dir, self.model_name, "train"))
        try:
            os.makedirs(self.training_dir, exist_ok=True)
        except OSError as err:
            BuildLogger.LOGGER.error(err)
            BuildLogger.LOGGER.info(access_hint("TRAINING_DIR"))
            sys.exit(1)

        # TODO - Document.
        self.training_file: str = os.getenv("TRAINING_FILE", "train.csv")
        # TODO - Document.
        self.training_path: str = os.path.join(self.training_dir, self.training_file)

        # TODO - Document.
        self.validation_dir: str = os.getenv("VALIDATION_DIR", os.path.join(self.data_dir, self.model_name, "validate"))
        try:
            os.makedirs(self.validation_dir, exist_ok=True)
        except OSError as err:
            BuildLogger.LOGGER.error(err)
            BuildLogger.LOGGER.info(access_hint("VALIDATION_DIR"))
            sys.exit(1)

        # TODO - Document.
        self.validation_file: str = os.getenv("VALIDATION_FILE", "validate.csv")
        # TODO - Document.
        self.validation_path: str = os.path.join(self.validation_dir, self.validation_file)

        # TODO - Document.
        self.evaluation_dir: str = os.getenv("EVALUATION_DIR", os.path.join(self.data_dir, self.model_name, "evaluate"))
        try:
            os.makedirs(self.evaluation_dir, exist_ok=True)
        except OSError as err:
            BuildLogger.LOGGER.error(err)
            BuildLogger.LOGGER.info(access_hint("EVALUATION_DIR"))
            sys.exit(1)

        # TODO - Document.
        self.evaluation_file: str = os.getenv("EVALUATION_FILE", "evaluate.csv")
        # TODO - Document.
        self.evaluation_path: str = os.path.join(self.evaluation_dir, self.evaluation_file)

        # TODO - Document.
        self.regression_dir: str = os.getenv("REGRESSION_DIR", os.path.join(self.data_dir, self.model_name, "regress"))
        try:
            os.makedirs(self.regression_dir, exist_ok=True)
        except OSError as err:
            BuildLogger.LOGGER.error(err)
            BuildLogger.LOGGER.info(access_hint("REGRESSION_DIR"))
            sys.exit(1)

        # TODO - Document.
        self.regression_file: str = os.getenv("REGRESSION_FILE", "regress.csv")
        # TODO - Document.
        self.regression_path: str = os.path.join(self.regression_dir, self.regression_file)

        # TODO - Document.
        self.log_dir: str = os.getenv("LOG_DIR", os.path.join(self.output_dir, "log", self.build_version))
        try:
            os.makedirs(self.log_dir, exist_ok=True)
        except OSError as err:
            BuildLogger.LOGGER.error(err)
            BuildLogger.LOGGER.info(access_hint("LOG_DIR"))
            sys.exit(1)

        # TODO - Document.
        try:
            self.validation_split: float = float(os.getenv("VALIDATION_SPLIT", "0.1"))
            if self.validation_split <= 0 or self.validation_split >= 1:
                raise ValueError("validation split must be a floating point value between 0-1 non-inclusive")
        except ValueError as err:
            BuildLogger.LOGGER.error(err)
            # TODO - Add intuitive error message.
            sys.exit(1)

        # TODO - Document.
        try:
            self.batch_size: int = int(os.getenv("BATCH_SIZE", "16"))
            if self.batch_size < 8 or self.batch_size > 512:
                raise ValueError("batch size must be an integer value between 8-512")
        except ValueError as err:
            BuildLogger.LOGGER.error(err)
            BuildLogger.LOGGER.info(
                "make sure the BATCH_SIZE environment variable is " +
                "an integer value between 8-512 inclusive")
            sys.exit(1)

        # TODO - Document.
        try:
            self.epochs: int = int(os.getenv("EPOCHS", "5"))
            if self.epochs < 0 or self.epochs > 1000:
                raise ValueError("epochs size must be an integer value between 0-1000")
        except ValueError as err:
            BuildLogger.LOGGER.error(err)
            BuildLogger.LOGGER.info(
                "make sure the EPOCHS environment variable is " +
                "an integer value between 0-1000 inclusive")
            sys.exit(1)
