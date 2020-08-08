"""
build_configuration.py - Module - The build_configuration.py module contains the Configuration class definition
which encapsulates logic related to configuring the build.
"""
import os
import datetime
import uuid


class BuildConfiguration:
    """
    BuildConfiguration - Class - The BuildConfiguration class encapsulates logic related to configuring
    the model builder.
    """
    MODEL_NAME = os.getenv("MODEL_NAME", uuid.uuid4())

    CREATE_DATASET_VISUALS = os.getenv("CREATE_DATASET_VISUALS", "1")
    try:
        CREATE_DATASET_VISUALS = int(CREATE_DATASET_VISUALS)
    except ValueError:
        CREATE_DATASET_VISUALS = 1

    CREATE_BUILD_VISUALS = os.getenv("CREATE_BUILD_VISUALS", "1")
    try:
        CREATE_BUILD_VISUALS = int(CREATE_BUILD_VISUALS)
    except ValueError:
        CREATE_BUILD_VISUALS = 1

    BASE_DIR = os.getenv("BASE_DIR", os.path.dirname(__file__))
    os.makedirs(BASE_DIR, exist_ok=True)

    BERT_DIR = os.getenv("BERT_DIR", os.path.join(BASE_DIR, "bert"))
    os.makedirs(BERT_DIR, exist_ok=True)

    DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
    os.makedirs(DATA_DIR, exist_ok=True)

    BUILD_VERSION = os.getenv("BUILD_VERSION", datetime.datetime.now().strftime("%Y%m%d-%H%M%s"))

    OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.join(BASE_DIR, "output"))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    BUILD_DIR = os.getenv("BUILD_DIR", os.path.join(OUTPUT_DIR, BUILD_VERSION))
    os.makedirs(BUILD_DIR, exist_ok=True)

    TESTING_DIR = os.getenv("TESTING_DIR", os.path.join(DATA_DIR, MODEL_NAME, "test"))
    os.makedirs(TESTING_DIR, exist_ok=True)

    TESTING_DATA = os.getenv("TESTING_DATA", os.path.join(TESTING_DIR, "test.csv"))

    TRAINING_DIR = os.getenv("TRAINING_DIR", os.path.join(DATA_DIR, MODEL_NAME, "train"))
    os.makedirs(TRAINING_DIR, exist_ok=True)

    TRAINING_DATA = os.getenv("TRAINING_DATA", os.path.join(TRAINING_DIR, "train.csv"))

    VALIDATION_DIR = os.getenv("VALIDATION_DIR", os.path.join(DATA_DIR, MODEL_NAME, "validate"))
    os.makedirs(VALIDATION_DIR, exist_ok=True)

    VALIDATION_DATA = os.getenv("VALIDATION_DATA", os.path.join(VALIDATION_DIR, "validate.csv"))

    REGRESSION_DIR = os.getenv("REGRESSION_DIR", os.path.join(DATA_DIR, MODEL_NAME, "regress"))
    os.makedirs(REGRESSION_DIR, exist_ok=True)

    REGRESSION_DATA = os.getenv("REGRESSION_DATA", os.path.join(REGRESSION_DIR, "regress.csv"))

    EVALUATION_DIR = os.getenv("EVALUATION_DIR", os.path.join(BUILD_DIR, "evaluation"))
    os.makedirs(EVALUATION_DIR, exist_ok=True)

    LOG_DIR = os.getenv("LOG_DIR", os.path.join(BASE_DIR, "log", BUILD_VERSION))
    os.makedirs(LOG_DIR, exist_ok=True)

    VALIDATION_SPLIT = os.getenv("VALIDATION_SPLIT", "0.1")
    try:
        VALIDATION_SPLIT = float(VALIDATION_SPLIT)
    except ValueError:
        VALIDATION_SPLIT = 0.1

    BATCH_SIZE = os.getenv("BATCH_SIZE", "16")
    try:
        BATCH_SIZE = int(BATCH_SIZE)
    except ValueError:
        BATCH_SIZE = 16

    EPOCHS = os.getenv("EPOCHS", "5")
    try:
        EPOCHS = int(5)
    except ValueError:
        EPOCHS = 5
