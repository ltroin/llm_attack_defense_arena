from ml_collections import config_dict

def get_config():
    config = config_dict.ConfigDict()

    # Define your configurations
    config.MAX_ALLOWED_ITERATION_PER_QUESTION = 75
    config.REPEAT_TIME_PER_QUESTION = 5

    return config
