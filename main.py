"""
Woodgate CLI (command line interface).
"""
import os
import uuid
import datetime
from absl import app
from absl import flags

from woodgate.woodgate_process import WoodgateProcess

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_name",
    str(uuid.uuid4()),
    """
    #: The `model_name` flag represents the name given to
    #: the machine learning evaluator. This attribute is set via
    #: the `--model_name` command line argument. If the
    #: `--model_name` command line argument. is not set, the
    #: `model_name` attribute is set to a random (v4) UUID by
    #: default.
    """
)

flags.DEFINE_string(
    "build_version",
    datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
    """
    #: The `build_version` attribute represents the specific
    #: version of the evaluator build_history. This attribute is
    #: set via the `--build_version` command line argument. If
    #: the `--command_line` command line argument is not set, the
    #: `build_version` attribute is set to a string formatted
    #: time ("%Y%m%d-%H%M%s") by default.
    """
)

flags.DEFINE_string(
    "woodgate_base_dir",
    os.path.join(
        os.path.expanduser("~"),
        "woodgate"
    ),
    """
    """
)

flags.DEFINE_string(
    "data_dir",
    os.path.join(
        os.path.expanduser("~"),
        "woodgate",
        "data"
    ),
    """
    """
)

flags.DEFINE_string(
    "output_dir",
    os.path.join(
        os.path.expanduser("~"),
        "woodgate",
        "output"
    ),
    """
    """
)

flags.DEFINE_string(
    "build_dir",
    os.path.join(
        os.path.expanduser("~"),
        "woodgate",
        "output",

    ),
    """
    """
)


def main(argv) -> None:
    """

    :return:
    :rtype:
    """
    WoodgateProcess.run()


if __name__ == "__main__":
    app.run(main)
