"""
dataset_retrieval_strategy.py - The
dataset_retrieval_strategy.py module contains the
DatasetRetrievalStrategy class definition.
"""
import gdown
from ..woodgate_settings import \
    WoodgateSettings


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
        training the learning model.

        :param url: URL string pointing to training dataset \
        located on Google Drive or Dropbox.
        :type url: str
        :param output: Path on host file system at which the \
        downloaded file will be saved.
        :type output: str
        :return: None
        :rtype: NoneType
        """
        gdown.download(url, output=output)

        return None
