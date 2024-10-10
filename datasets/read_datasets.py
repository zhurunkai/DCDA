import os


def return_dataset_paths(config):
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    server_dict = {
        "default": {
            "mit-states": os.path.join(DIR_PATH, "../data/mit-states"),
            "ut-zappos": os.path.join(DIR_PATH, "../data/ut-zappos"),
            "cgqa": os.path.join(DIR_PATH, "../data/cgqa")
        }
    }

    return server_dict['default']
