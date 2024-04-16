import os


def return_dataset_paths(config):
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))

    DATASET_PATHS = {
        "mit-states": os.path.join(DIR_PATH, "../DATA_ROOT/mit-states"),
        "ut-zappos": os.path.join(DIR_PATH, "../DATA_ROOT/ut-zappos"),
        "cgqa": os.path.join(DIR_PATH, "../DATA_ROOT/cgqa")
    }

    return DATASET_PATHS
