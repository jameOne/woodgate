"""
dataset_retrieval_strategy.py - The
dataset_retrieval_strategy.py module contains the
DatasetRetrievalStrategy class definition.
"""
import os
import uuid
import subprocess
from typing import Tuple
from ..woodgate_settings import FileSystem


class DatasetRetrievalStrategy:
    """
    DatasetRetrievalStrategy - The
    DatasetRetrievalStrategy class encapsulates logic
    related to collecting data required for fine tuning.
    """

    @staticmethod
    def retrieve_dataset(
            file_system: FileSystem,
            file_id: str,
            output: str
    ) -> Tuple[bytes, bytes]:
        """This method will retrieve the dataset residing at the
        provided `URL` (:param url:) and save to `output`
        (:param output:). The implementation currently uses
        `gdown` and so it is implied the URL be pointing to a
        file located in either Google Drive or Dropbox.
        What makes this method specific to the training dataset
        is that the output is automatically stored on the host
        file system at `$TRAINING_PATH`. When the
        `woodgate.woodgate_process.WoodgateProcess.run()` method
        is executed there is an assumption made related to file
        structure and calling this method to retrieve the
        training dataset ensures the correct file will be used
        for training the learning evaluator.

        :param file_system: An object representing a writable \
        file system configuration.
        :type file_system: FileSystem
        :param file_id: ID string pointing to training dataset \
        located on Google Drive or Dropbox.
        :type file_id: str
        :param output: Path on host file system at which the \
        downloaded file will be saved.
        :type output: str
        :return: Tuple[bytes, bytes]
        :rtype: Tuple
        """
        random_uuid = uuid.uuid4()
        cookie_path = os.path.join(
            file_system.temp_dir,
            str(random_uuid)
        )
        sh = [
            f'curl -sc {cookie_path} "https://drive.google.com/uc'
            + f'?export=download&id={file_id}" > /dev/null',
            f'code="$(awk \'/_warning_/ {{print $NF}}\' {cookie_path})"',
            f'curl -Lb {cookie_path} "https://drive.google.com/uc'
            + f'?export=download&confirm=${{code}}&id={file_id}" '
            + f'-o {output}'
        ]

        process = subprocess.Popen(
            "; ".join(sh),
            executable='/bin/bash',
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        stdout, stderr = process.communicate()

        return stdout, stderr
