"""
build_summary.py - Module - The build_summary.py module contains the BuildSummary class definition
which encapsulates logic related to summarizing the build.
"""
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from build_configuration import BuildConfiguration


class BuildSummary:
    """
    BuildSummary - Class - The BuildSummary class encapsulates logic related to summarizing the
    model build.
    """

    @staticmethod
    def create_loss_over_epochs_plot(build_history):
        """

        :param build_history:
        :type build_history:
        :return:
        :rtype:
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
        plt.savefig(os.path.join(
            BuildConfiguration.OUTPUT_DIR, "post_train_summary", "loss_over_epochs.png"))
        plt.figure().clear()

    @staticmethod
    def create_accuracy_over_epochs_plot(build_history):
        """
        
        :param build_history: 
        :type build_history: 
        :return: 
        :rtype: 
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
        plt.savefig(os.path.join(
            BuildConfiguration.OUTPUT_DIR, "post_train_summary", "accuracy_over_epochs.png"))
        plt.figure().clear()
