"""
build_logger.py - The build_logger.py module contains the BuildLogger class definition.
"""
import logging


class BuildLogger:
    """
    BuildLogger - The BuildLogger class encapsulates logic related defining the shared logger utility.
    """
    logging.basicConfig(level=logging.DEBUG)
    LOGGER = logging.getLogger("build_logger")
