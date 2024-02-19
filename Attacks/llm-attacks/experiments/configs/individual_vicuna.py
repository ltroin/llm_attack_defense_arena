import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.result_prefix = '../../gcg'

    config.tokenizer_paths=["../../../models/lmsys/vicuna-7b-v1.5"]
    config.model_paths=["../../../models/lmsys/vicuna-7b-v1.5"]
    config.conversation_templates=['vicuna']
    config.success_count = 50
    config.model_name = 'vicuna'
    return config