"""
file_system_configuration.py - The file_system_configuration.py
module contains the FileSystemConfiguration class definition.
"""
import os
from typing import Type
from ..woodgate_settings import WoodgateSettings


class FileSystemConfiguration:
    """
    FileSystemConfiguration - The FileSystemConfiguration class
    encapsulates logic related to configuring the model builder.
    """

    def __init__(
            self,
            woodgate_settings: Type[WoodgateSettings]
    ):
        """The __init__ method assigns *_dir attributes from
        the `woodgate_settings: WoodgateSettings` argument
        to the object, then calls `self.configure_file_system()`
        which attempts to create all the directories. If the
        directory exists the error is ignored.

        :param woodgate_settings: The Woodgate process \
        configuration.
        :type woodgate_settings: WoodgateSettings
        """
        self.woodgate_base_dir: str = woodgate_settings\
            .woodgate_base_dir
        self.data_dir: str = woodgate_settings.data_dir
        self.output_dir: str = woodgate_settings.output_dir
        self.build_dir: str = woodgate_settings.build_dir
        self.model_build_dir: str = woodgate_settings\
            .model_build_dir
        self.model_data_dir: str = woodgate_settings\
            .model_data_dir
        self.log_dir: str = woodgate_settings.log_dir
        self.training_dir: str = woodgate_settings.training_dir
        self.testing_dir: str = woodgate_settings.testing_dir
        self.evaluation_dir: str = woodgate_settings\
            .evaluation_dir
        self.regression_dir: str = woodgate_settings\
            .regression_dir
        self.datasets_summary_dir: str = woodgate_settings\
            .datasets_summary_dir
        self.build_summary_dir: str = woodgate_settings\
            .build_summary_dir
        self.evaluation_summary_dir: str = woodgate_settings\
            .evaluation_summary_dir
        self.bert_dir: str = woodgate_settings.bert_dir
        self.configure_file_system()

    def configure_file_system(self) -> None:
        """This method will iterate over the instance's
        attributes and pass the

        :return: None
        :rtype: NoneType
        """
        for attr, val in self.__dict__.items():
            os.makedirs(val, exist_ok=True)

        return None
