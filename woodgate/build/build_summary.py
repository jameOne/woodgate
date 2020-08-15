"""
build_summary.py - The build_summary.py module contains the
BuildSummary class definition.
"""
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from .file_system_configuration import FileSystemConfiguration
import tensorflow as tf


class BuildSummary:
    """
    BuildSummary - The BuildSummary class encapsulates logic
    related to summarizing the model build.
    """

    @staticmethod
    def create_loss_over_epochs_plot(
        build_history: tf.keras.callbacks.History
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
        :type build_history: object
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
                FileSystemConfiguration.output_dir,
                "build_summary",
                "loss_over_epochs.png"
            )
        )
        plt.figure().clear()

        return None

    @staticmethod
    def create_accuracy_over_epochs_plot(
            build_history: tf.keras.callbacks.History
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
        :type build_history: object
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
                FileSystemConfiguration.output_dir,
                "build_summary",
                "accuracy_over_epochs.png"
            )
        )
        plt.figure().clear()

        return None
