"""
build_configuration.py - The build_configuration.py module
contains the BuildConfiguration class definition.
"""
import os
import datetime
import uuid


class BuildConfiguration:
    """
    BuildConfiguration - The BuildConfiguration class encapsulates
    logic related to configuring the model builder.
    """

    def __init__(self):
        #: The `model_name` attribute represents the name given to
        #: the machine learning model. This attribute is set via
        #: the `MODEL_NAME` environment variable. If the
        #: `MODEL_NAME` environment variable is not set, the
        #: `model_name` attribute is set to a random (v4) UUID by
        #: default.
        self.model_name: str = os.getenv(
            "MODEL_NAME",
            str(uuid.uuid4())
        )

        #: The `build_version` attribute represents the specific
        #: version of the model build. This attribute is set via
        #: the `BUILD_VERSION` environment variable. If the
        #: `BUILD_VERSION` environment variable is not set, the
        #: `build_version` attribute is set to a string formatted
        #: time ("%Y%m%d-%H%M%s") by default.
        self.build_version: str = os.getenv(
            "BUILD_VERSION",
            datetime.datetime.now().strftime("%Y%m%d-%H%M%s")
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
        self.create_dataset_visuals: int = int(
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
        self.create_build_visuals: int = int(
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
        self.woodgate_base_dir: str = os.getenv(
            "WOODGATE_BASE_DIR",
            "~/woodgate"
        )
        os.makedirs(self.woodgate_base_dir, exist_ok=True)

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
        self.data_dir: str = os.getenv("DATA_DIR", os.path.join(
            self.woodgate_base_dir, "data"))
        os.makedirs(self.data_dir, exist_ok=True)

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
        self.output_dir: str = os.getenv(
            "OUTPUT_DIR",
            os.path.join(
                self.woodgate_base_dir,
                "output"
            )
        )
        os.makedirs(self.output_dir, exist_ok=True)

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
        self.build_dir: str = os.getenv(
            "BUILD_DIR",
            os.path.join(
                self.output_dir,
                self.build_version
            )
        )
        os.makedirs(self.build_dir, exist_ok=True)

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
        self.model_build_dir: str = os.getenv(
            "MODEL_BUILD_DIR",
            os.path.join(
                self.build_dir,
                self.model_name
            )
        )
        os.makedirs(self.output_dir, exist_ok=True)

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
        self.model_data_dir: str = os.getenv(
            "MODEL_DATA_DIR",
            os.path.join(
                self.data_dir,
                self.model_name
            )
        )
        os.makedirs(self.output_dir, exist_ok=True)
