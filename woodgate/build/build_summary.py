"""
build_summary.py - The build_summary.py module contains the
BuildSummary class definition.
"""
import os
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from ..woodgate_settings import WoodgateSettings
from tensorflow import keras


class BuildSummary:
    """
    BuildSummary - The BuildSummary class encapsulates logic
    related to summarizing the model build.
    """

    @staticmethod
    def create_loss_over_epochs_json(
            build_history: keras.callbacks.History
    ) -> None:
        """The `create_loss_over_epochs_json` method creates a
        json document on the host file system in the
        `WoodgateSettings.build_summary_dir` directory.

        :param build_history:
        :type build_history:
        :return: None
        :rtype: NoneType
        """
        loss_over_epochs_dict = {
            "loss": build_history.history['loss'],
            "valLoss": build_history.history['val_loss'],
            "title": 'Loss over training epochs',
            "yLabel": 'Loss',
            "xLabel": 'Epoch',
            "legend": ['train', 'test']
        }
        loss_over_epochs_json = os.path.join(
                WoodgateSettings.build_summary_dir,
                "lossOverEpochs.json"
            )

        with open(loss_over_epochs_json, "w+") as file:
            file.write(
                json.dumps(loss_over_epochs_dict)
            )

        return None

    @staticmethod
    def create_loss_over_epochs_plot(
        build_history: keras.callbacks.History
    ) -> None:
        """This method will generate an Loss vs. Epochs plot
        from the `tf.keras.callbacks.History` object. The
        generated plot is in PNG format having `.png` file
        extension and will be located at
        `$OUTPUT_DIR/build_summary/loss_over_epochs.png` on
        host file system.

        :param build_history: Accepts a History object, where \
        the History object is the return type of calling the \
        `fit` method on `tf.keras.Model` objects.
        :type build_history: keras.callbacks.History
        :return: None
        :rtype: NoneType
        """

        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.plot(build_history.history['loss'])
        ax.plot(build_history.history['val_loss'])
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'])
        plt.title('Loss over training epochs')
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                WoodgateSettings.build_summary_dir,
                "loss_over_epochs.png"
            )
        )
        plt.clf()

        return None

    @staticmethod
    def create_accuracy_over_epochs_json(
            build_history: keras.callbacks.History
    ) -> None:
        """

        :param build_history:
        :type build_history:
        :return:
        :rtype:
        """
        acc_over_epochs_dict = {
            "acc": build_history.history['acc'],
            "valLoss": build_history.history['val_acc'],
            "title": 'Accuracy over training epochs',
            "yLabel": 'Accuracy',
            "xLabel": 'Epoch',
            "legend": ['train', 'test']
        }
        acc_over_epochs_json = os.path.join(
            WoodgateSettings.build_summary_dir,
            "accuracyOverEpochs.json"
        )

        with open(acc_over_epochs_json, "w+") as file:
            file.write(
                json.dumps(acc_over_epochs_dict)
            )

        return None

    @staticmethod
    def create_accuracy_over_epochs_plot(
            build_history: keras.callbacks.History
    ) -> None:
        """This method will generate an Accuracy vs. Epochs plot
        from the `tf.keras.callbacks.History` object. The
        generated plot is in PNG format having `.png` file
        extension and will be located at
        `$OUTPUT_DIR/build_summary/accuracy_over_epochs.png` on
        host file system.
        
        :param build_history: Accepts a `History` object, where \
        the `History` object is the return type of calling the \
        `fit` method on `keras.Model` objects.
        :type build_history: keras.callbacks.History
        :return: None
        :rtype: NoneType
        """

        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.plot(build_history.history['acc'])
        ax.plot(build_history.history['val_acc'])
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'])
        plt.title('Accuracy over training epochs')
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                WoodgateSettings.build_summary_dir,
                "accuracy_over_epochs.png"
            )
        )
        plt.clf()

        return None
