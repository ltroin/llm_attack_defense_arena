import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.result_prefix = '../../gcg'

    config.tokenizer_paths=["../../../models/meta-llama/Llama-2-7b-chat-hf"]
    config.model_paths=["../../../models/meta-llama/Llama-2-7b-chat-hf"]
    config.conversation_templates=['llama-2']
    config.success_count = 50
    config.model_name = 'llama-2'
    return config