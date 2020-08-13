"""
woodgate_logger.py - The woodgate_logger.py module contains the
BuildLogger class definition.
"""
import os
import logging
from woodgate.build.build_configuration import BuildConfiguration


class WoodgateLogger:
    """
    BuildLogger - The BuildLogger class encapsulates logic related
    to defining the shared logger utility.
    """
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("build_logger")

    def __init__(self, build_configuration: BuildConfiguration):
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
        self.log_dir: str = os.getenv(
            "LOG_DIR",
            os.path.join(
                build_configuration.output_dir,
                "log",
                build_configuration.build_version
            )
        )
        os.makedirs(self.log_dir, exist_ok=True)

        #: The `log_file` attribute represents the base name of
        #: the `log_path` attribute. The log file should therefore
        #: reside in the `log_dir` by definition. This file should
        #: have a CSV file (having a `.csv` file extension). This
        #: attribute is set via the `REGRESSION_FILE` environment
        #: variable. If the `LOG_FILE` environment variable is not
        #: set, then the `log_file` will default to
        #: `$BUILD_VERSION.log`.
        self.log_file: str = os.getenv(
            "LOG_FILE",
            f"{build_configuration.build_version}.log"
        )

        #: The `log_path` attribute represents the full path on
        #: the host file system pointing to `log_file`. This
        #: attribute is set via the `LOG_PATH` environment
        #: variable. If `LOG_PATH` is set, then it will render
        #: values set by `log_dir` and `log_file` inconsequential.
        #: If  `LOG_PATH` is not set, then the `log_path` will
        #: default to `$LOG_DIR/$LOG_FILE`.
        self.log_path: str = os.path.join(
            self.log_dir,
            self.log_file
        )
