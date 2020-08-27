"""
Woodgate CLI (command line interface).
"""
from absl import app
from absl import flags

from woodgate.woodgate_process import WoodgateProcess
from woodgate.woodgate_settings import (
    Architecture,
    Build,
    FileSystem,
    Model
)

FLAGS = flags.FLAGS

# Model
flags.DEFINE_string(
    "model_name",
    "woodgate_model",
    """
    #: The `model_name` flag represents the name given to
    #: the machine learning evaluator. This attribute is set
    #: via the `--model_name` command line argument. If the
    #: `--model_name` command line argument. is not set, the
    #: `model_name` attribute is set to a random (v4) UUID by
    #: default.
    """
)

flags.DEFINE_string(
    "model_uuid",
    "",
    """
    #: The `--model_uuid` flag represents the unique identity
    #: given to the machine learning model. A random (v4) UUID
    #: will be assigned if the `--model_uuid` is not supplied.
    #: If the user supplies a UUID, but the supplied UUID does
    #: not exist (correspond to a known model) the value will
    #: be disregarded and a new, valid, UUID will be generated
    #: for the model.
    """
)

# Architecture
flags.DEFINE_float(
    "clf_out_dropout_rate",
    0.5,
    """
    #: The `--clf_out_dropout_rate` flag represents one
    #: of two (1 / 2) dropout rates which may be customized.
    #: Typically this value should be around `0.5`. If the
    #: `--clf_out_dropout_rate` flag is not set, then
    #: `--clf_out_dropout_rate` defaults to `0.5`.
    """,
    lower_bound=0.0,
    upper_bound=1.0
)

flags.DEFINE_enum(
    "clf_out_activation",
    "tanh",
    Architecture.ACTIVATIONS,
    """
    #: The `--clf_out_activation` flag represents one of two
    #: (1 / 2) activation functions which may be customized.
    #: If the `--clf_out_activation` flag is not set, then
    #: `--clf_out_activation` defaults to `tanh`.
    """
)

flags.DEFINE_float(
    "logits_dropout_rate",
    0.5,
    """
    #: The `--logits_dropout_rate` flag represents two
    #: of two (2 / 2) dropout rates which may be customized.
    #: Typically this value should be around `0.5`. If the
    #: `--logits_dropout_rate` flag is not set, then
    #: `--logits_dropout_rate` defaults to `0.5`.
    """,
    lower_bound=0.0,
    upper_bound=1.0
)

flags.DEFINE_enum(
    "logits_activation",
    "softmax",
    Architecture.ACTIVATIONS,
    """
    #: The `--logits_activation` flag represents two of two
    #: (2 / 2) activation functions which may be customized.
    #: If the `--logits_activation` flag is not set, then
    #: `--logits_activation` defaults to `softmax`.
    """
)

# Trainer
flags.DEFINE_float(
    "validation_split",
    0.1,
    """
    #: The `--validation_split` flag represents a decimal
    #: number between 0 and 1 inclusive. Validation split
    #: indicates the proportional split of your training set
    #: by the value of the variable. For example, a value of
    #: `--validation_split 0.2` would signal the program to
    #: reserve 20% of the training set for validation testing
    #: completed after each training epoch. If the
    #: `--validation_split` flag is not set, then 
    #: `--validation_split` will default to `0.1`.
    """,
    lower_bound=0.0,
    upper_bound=1.0
)

flags.DEFINE_integer(
    "batch_size",
    16,
    """
    #: The `--batch_size` flag represents an integer number
    #: between 1 and 1024 inclusive. This value indicates the
    #: number of training examples utilized in one iteration.
    #: The batch size is a characteristic of gradient descent
    #: training algorithms. If the `--batch_size` flag is not
    #: set, then the `--batch_size` attribute will default to
    #: `16`.
    """,
    lower_bound=1,
    upper_bound=1024
)

flags.DEFINE_integer(
    "epochs",
    1,
    """
    #: The `--epochs` flag represents an integer between
    #: 1 and 1024 inclusive. This value indicates the number of
    #: times the training algorithm will iterate over the
    #: training dataset before completing. If the `--epochs`
    #: environment variable is unset, then `--epochs`
    #: will default to `1`.
    """,
    lower_bound=1,
    upper_bound=1024
)

flags.DEFINE_boolean(
    "log_tensorboard",
    True,
    """
    #: The `--log_tensorboard` flag represents a boolean value
    #: This value indicates whether or not TensorBoard logs will
    #: be generated during training. If the `--log_tensorboard`
    #: environment variable is unset, then `--log_tensorboard`
    #: will default to `True`.
    """
)


def main(argv) -> None:
    """

    :return:
    :rtype:
    """
    if argv[1] == "run":
        model = Model(
            model_name=FLAGS.model_name,
            model_uuid=FLAGS.model_uuid
        )
        build = Build()
        file_system = FileSystem(model, build)
        file_system.configure()

        WoodgateProcess.run(model=model, file_system=file_system)


if __name__ == "__main__":
    app.run(main)
