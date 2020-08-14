"""
woodgate_logger.py - The woodgate_logger.py module contains the
BuildLogger class definition.
"""
import os
import logging
from woodgate.build.file_system_configuration import FileSystemConfiguration


class WoodgateLogger:
    """
    BuildLogger - The BuildLogger class encapsulates logic related
    to defining the shared logger utility.
    """
    logging.basicConfig(level=logging.DEBUG)

    #: The `logger` attribute represents the logger instance which
    #: should be used throughout the Woodgate process.
    logger = logging.getLogger("build_logger")
