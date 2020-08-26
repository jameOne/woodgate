"""
woodgate_logger.py - The woodgate_logger.py module contains the
BuildLogger class definition.
"""
import logging
from .woodgate_settings import FileSystem


class WoodgateLogger:
    """
    BuildLogger - The BuildLogger class encapsulates logic
    related to defining the shared logger utility.
    """

    def __init__(self, file_system: FileSystem):
        """

        :param file_system:
        :type file_system:
        """
        #: The `logger` attribute represents the logger instance
        #: which should be used throughout the Woodgate process.
        logger = logging.getLogger("build_logger")
        logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(file_system.get_log_path())
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        # create formatter and add it to the handlers
        formatter = logging.Formatter(
            '%(asctime)s -'
            + '%(name)s -'
            + '%(levelname)s -'
            + '%(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)
        self.logger = logger
