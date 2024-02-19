
from src.framework.framework_model import FrameworkModel
import os
import sys
parent_parent_dir = os.path.abspath(os.path.join(os.getcwd(), '../../../../'))
import models
import tiktoken
class Primary(FrameworkModel):
    """The primary model.  Takes user input and generates a response as normal"""

    def __init__(self, model):
        """
        Args:
            model_info: The model information for the primary model"""

        self.model = "11"

    @classmethod
    def from_model_name(cls, primary_model_path: str):
        """Creates a primary model from its name

        Args:
            primary_model_name: The name of the primary model
            model_src: The suggested source of the model to load. Defaults to AUTO
        Returns:
            An instance of a primary model"""
        if 'gpt' in primary_model_path:
            model = models.OpenAILLM(model_path=primary_model_path)
            tokenizer = "None"
        else:
            modelObj= models.LocalVLLM(model_path=primary_model_path)

        

        return cls(p_model_info)

    @property
    def name(self):
        return f"P({self.model.name_or_path})"

    def generate(self, prompt: str,params=None, **kwargs):
        return self.generate_using(prompt, self.model, self.tokenizer,params, **kwargs)
