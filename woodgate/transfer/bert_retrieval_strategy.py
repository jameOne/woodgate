"""
bert_retrieval_strategy.py - This module contains the
BertRetrievalStrategy class definition.
"""
import os
import subprocess
from ..build.file_system_configuration import \
    FileSystemConfiguration
from ..woodgate_logger import WoodgateLogger
from .bert_model_parameters import BertModelParameters


class BertRetrievalStrategy:
    """
    BertRetrievalStrategy - This class encapsulates logic
    related to downloading Google's BERT for transfer learning.
    """
    #: The `bert_model_parameters` attribute
    bert_model_parameters = BertModelParameters()

    #: The `bert_base_url` attribute represents the base
    #: URL/endpoint where the BERT model is hosted. This
    #: attribute is set via the `BERT_BASE_URL` environment
    #: variable. If the `BERT_BASE_URL` environment variable is
    #: not set, then `bert_base_url` defaults to
    #: `https://storage.googleapis.com/`.
    #:
    #: See `https://github.com/google-research/bert` for more on
    #: where BERT is hosted.
    bert_base_url: str = os.getenv(
        "BERT_BASE_URL",
        'https://storage.googleapis.com'
    )

    #: The `bert_models_url_component` is the URL component
    #: directly preceded by `bert_base_url`. The
    #: `bert_models_url_component` is set via the
    #: `BERT_MODELS_URL_COMPONENT` environment variable.
    #: If the `BERT_MODELS_URL_COMPONENT` environment variable is
    #: not set, then the `bert_models_url_component` default to
    #: the string `bert_models`.
    #:
    #: See `https://github.com/google-research/bert` for more on
    #: where BERT is hosted.
    bert_models_url_component: str = os.getenv(
        "BERT_MODELS_URL_COMPONENT",
        "bert_models"
    )

    #: The `bert_models_url` attribute represents the URL to the
    #: bert models directory. This attribute is set via
    #: the `BERT_MODELS_URL` environment variable. If the
    #: `BERT_MODELS_URL` environment variable is not set, then
    #: the `bert_models_url` defaults to
    #: `bert_base_url/bert_models_url_component`.
    #:
    #: See `https://github.com/google-research/bert` for more on
    #: where BERT is hosted.
    bert_models_url: str = os.getenv(
        "BERT_MODELS_URL",
        os.path.join(
            bert_base_url,
            bert_models_url_component
        )
    )

    #: The `bert_version_url_component` is the URL component
    #: directly preceded by `bert_models_url`. The
    #: `bert_version_url_component` is set via the
    #: `BERT_MODELS_URL_COMPONENT` environment variable.
    #: If the `BERT_MODELS_URL_COMPONENT` environment variable is
    #: not set, then the `bert_version_url_component` default to
    #: the string `2020_02_20`.
    #:
    #: See `https://github.com/google-research/bert` for more on
    #: where BERT is hosted.
    bert_version_url_component: str = os.getenv(
        "BERT_VERSION_URL_COMPONENT",
        "2020_02_20"
    )

    #: The `bert_version_url` attribute represents the URL to the
    #: bert version directory. This attribute is set via
    #: the `BERT_VERSION_URL` environment variable. If the
    #: `BERT_VERSION_URL` environment variable is not set, then
    #: the `bert_version_url` defaults to
    #: `bert_models_url/bert_version_url_component`.
    #:
    #: See `https://github.com/google-research/bert` for more on
    #: where BERT is hosted.
    bert_version_url: str = os.getenv(
        "BERT_VERSION_URL",
        os.path.join(
            bert_models_url,
            bert_version_url_component
        )
    )

    #: The `bert_zip_url_component` is the URL component
    #: directly preceded by `bert_version_url`. The
    #: `bert_zip_url_component` is set via the
    #: `BERT_ZIP_URL_COMPONENT` environment variable.
    #: If the `BERT_ZIP_URL_COMPONENT` environment variable is
    #: not set, then the `bert_zip_url_component` defaults to
    #: the string `uncased_L-12_H-768_A-12.zip` (BERT base).
    #:
    #: See `https://github.com/google-research/bert` for more on
    #: where BERT is hosted.
    bert_zip_url_component: str = os.getenv(
        "BERT_ZIP_URL_COMPONENT",
        "uncased_"
        + f"L-{bert_model_parameters.bert_l_param}_"
        + f"H-{bert_model_parameters.bert_h_param}_"
        + f"A-{bert_model_parameters.bert_a_param}.zip"
    )

    #: The `bert_zip_url` attribute represents the URL to the
    #: bert zip directory. This attribute is set via
    #: the `BERT_ZIP_URL` environment variable. If the
    #: `BERT_ZIP_URL` environment variable is not set, then
    #: the `bert_zip_url` defaults to
    #: `bert_version_url/bert_zip_url_component`.
    #:
    #: See `https://github.com/google-research/bert` for more on
    #: where BERT is hosted.
    bert_zip_url: str = os.getenv(
        "BERT_ZIP_URL",
        os.path.join(
            bert_version_url,
            bert_zip_url_component
        )
    )

    @classmethod
    def download_bert(cls) -> None:
        """

        :return: None
        :rtype: NoneType
        """
        print(cls.bert_zip_url)
        # downloading the models is an expensive process
        # so first make sure we actually need the files
        if not os.path.isfile(
                FileSystemConfiguration.bert_model_path
        ) or not os.path.isfile(
            FileSystemConfiguration.bert_config_path
        ) or not os.path.isfile(
            FileSystemConfiguration.bert_vocab_path
        ):
            process = subprocess.Popen(
                [
                    'wget',
                    cls.bert_zip_url
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
            WoodgateLogger.logger.info(stdout)
            WoodgateLogger.logger.error(stderr)

            process = subprocess.Popen(
                [
                    'unzip',
                    cls.bert_zip_url_component,
                    '-d',
                    FileSystemConfiguration.bert_dir
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
            WoodgateLogger.logger.info(stdout)
            WoodgateLogger.logger.error(stderr)

            process = subprocess.Popen(
                [
                    'rm',
                    f'{cls.bert_zip_url_component}'
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
            WoodgateLogger.logger.info(stdout)
            WoodgateLogger.logger.error(stderr)

        return None
