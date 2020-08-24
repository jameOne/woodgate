"""
dataset_retrieval_strategy.py - The
dataset_retrieval_strategy.py module contains the
DatasetRetrievalStrategy class definition.
"""
import os
import stat
from urllib.parse import urlparse
from typing import List
import subprocess
from ..woodgate_logger import WoodgateLogger


class DatasetRetrievalStrategy:
    """
    DatasetRetrievalStrategy - The
    DatasetRetrievalStrategy class encapsulates logic
    related to collecting data required for fine tuning.
    """

    @staticmethod
    def retrieve_tuning_dataset(
            url: str,
            output: str
    ) -> None:
        """This method will retrieve the dataset residing at the
        provided `URL` (:param url:) and save to `output`
        (:param output:). The implementation currently uses `gdown`
        and so it is implied the URL be pointing to a
        file located in either Google Drive or Dropbox.
        What makes this method specific to the training dataset is
        that the output is automatically stored on the host file
        system at `$TRAINING_PATH`. When the
        `woodgate.woodgate_process.WoodgateProcess.run()` method
        is executed there is an assumption made related to file
        structure and calling this method to retrieve the training
        dataset ensures the correct file will be used for
        training the learning evaluator.

        :param url: URL string pointing to training dataset \
        located on Google Drive or Dropbox.
        :type url: str
        :param output: Path on host file system at which the \
        downloaded file will be saved.
        :type output: str
        :return: None
        :rtype: NoneType
        """

        parsed_url = urlparse(url)
        query_params = parsed_url.query
        query_lol = [
            query.split("=") for query in query_params.split("&")
        ]

        file_id = ""
        for ls in query_lol:
            if ls[0] == 'id':
                file_id = ls[1]

        os.environ["FILE_ID"] = file_id
        os.environ["DOWNLOAD_FILE_NAME"] = output

        sh = [
            f'fileId={file_id} && ' + f'fileName={output} && ',
            'curl -sc ./cookie "https://drive.google.com/uc?export'
            + '=download&id=${fileId}" > /dev/null\n',
            'code="$(awk \'/_warning_/ {print $NF}\' ./cookie)"\n',
            'curl -Lb ./cookie "https://drive.google.com/uc?export'
            + '=download&confirm=${code}&id=${fileId}" -o ${fileName}\n',
            'rm ./cookie\n']

        with open("./download.sh", "w+") as file:
            file.writelines(sh)
            st = os.stat("./download.sh")
            os.chmod(
                "./download.sh", st.st_mode | stat.S_IEXEC)

        process = subprocess.Popen(
            ['/bin/bash', './download.sh'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()

        if stdout:
            WoodgateLogger.logger.info(stdout)

        if stderr:
            WoodgateLogger.logger.error(stderr)

        os.remove("./download.sh")

        return None
