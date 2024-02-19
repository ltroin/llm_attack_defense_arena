import gc
import numpy as np
import torch
import torch.nn as nn
import time
import argparse
import os
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
import subprocess
from prompt_toolkit.shortcuts import ProgressBar
import re
import gc
import logging
from download_models import download

logging.basicConfig(level=logging.INFO)
os.environ["MKL_THREADING_LAYER"] = "GNU"


def _clear_memo():
    tensors_to_delete = []
    
    for obj in gc.get_objects():
        
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(obj)
                tensors_to_delete.append(obj)
        except:
            pass
    
    for tensor in tensors_to_delete:
        # print("delete",tensor)
        del tensor
    
    gc.collect()
    torch.cuda.empty_cache()


def _init_models(name):
    download(name)
    

def args_to_cmd(args):
    cmd = []
    for key, value in args.items():
        cmd.append(f"--{key}")
        cmd.append(str(value))
    return cmd

class BaseAttackModel:
    def __init__(self):
        self.parameters = {}

    def collect_parameters(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def Execute(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

# This is a white box attack by method, you could not use it for GPT
class AutoDAN(BaseAttackModel):

    def __init__(self,model):
        super().__init__()
        self.parameters = {
            'model': model,
            'num_steps': 100,
            'batch_size': 64,
            'num_elites': 0.1,
            'crossover': 0.5,
            'num_points': 5,
            'mutation': 0.01,
            'low_memory': 1,
            'num_success_prompts': 1000, # no effect
        }


    def run(self):
        print(f"Running AUTODAN with model: {self.parameters['model']}")
       # Define the command and arguments
        command = "python"
        script = "AutoDAN.py"
        args = args_to_cmd(self.parameters)

        # Combine the command and arguments into a single list
        cmd =  ['stdbuf', '-oL'] + [command,'-u', script] + args
        
        # Start the subprocess and capture its output
        try:
            with subprocess.Popen(cmd, cwd="./Attacks/AutoDAN/start", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,text=True,bufsize=1) as sp:
                for line in sp.stdout:
                    logging.info(line)
        finally:
            sp.terminate()
            sp.wait()

# # this is a white box attack
# class GCG(BaseAttackModel):
#     def __init__(self):
#         # Set up default parameters
#         super().__init__()
#         self.parameters = {
#             'config': "./configs/individual_llama2.py",  # you can change it to individual_vicuna
#             'config.attack': 'gcg',
#             'config.train_data': "../data/advbench/harmful_behaviors.csv",
#             'config.result_prefix': "../../../individual",
#             'config.n_train_data': 15,
#             'config.n_steps': 300,
#             'config.test_steps': 100,
#             'config.batch_size': 512,
#             'config.success_count' : 5,
#             'config.topk' : 256,
#             'config.filter_cand' :True
#         }

#         # Override defaults with any provided argparse values
#         # self.__dict__.update({k: v for k, v in vars(args).items() if v is not None})

#     def run(self):
#         print("Running main.py GCG with specified parameters")
#         command = "python"
#         script = "main.py"
#         args = args_to_cmd(self.parameters)

#         cmd = ['stdbuf', '-oL', command, '-u', script] + args

#         # Start the subprocess and capture its output
#         try:
#             with subprocess.Popen(cmd, cwd="./Attacks/llm-attacks/experiments", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as sp:
#                 for line in sp.stdout:
#                     logging.info(line)
#         finally:
#             sp.terminate()
#             sp.wait()
            
class GCG(BaseAttackModel):
    def __init__(self,model):
        # Set up default parameters
        super().__init__()
        self.parameters = {
            'model_path':model,
            'num_steps': 500,
            'batch_size': 512,
            'topk' : 256,
            'filter_cand' :True,
            'max_success' : 1024,
        }

    def run(self):
        print("Running test.py GCG with specified parameters")
        command = "python"
        script = "test.py"
        args = args_to_cmd(self.parameters)

        cmd = ['stdbuf', '-oL', command, '-u', script] + args

        # Start the subprocess and capture its output
        try:
            with subprocess.Popen(cmd, cwd="./Attacks/llm-attacks", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as sp:
                for line in sp.stdout:
                    logging.info(line)
        finally:
            sp.terminate()
            sp.wait()

class GPTFuzz(BaseAttackModel):
    def __init__(self,model):
        # Set up default parameters
        super().__init__()
        self.parameters = {
            'model_path': 'gpt-3.5-turbo', #only openai models,used for attack
            # 'target_model': "../../models/meta-llama/Llama-2-7b-chat-hf", #[gpt-3.5-turbo', 'gpt-4-turbo', llamaPath, VicunaPath]
            'target_model': model, #[gpt-3.5-turbo', 'gpt-4-turbo', llama, vicuna]
            'max_query': 10,
            # 'max_jailbreak': 1,
            'energy': 1,
            # 'seed_selection_strategy': 50,
            # 'max-new-tokens': 512,
            'seed_path' : "./datasets/prompts/GPTFuzzer.csv",
        }


    def run(self):
        print("Running main.py with specified parameters for GPTFUZZ")
        command = "python"
        script = "gptfuzz.py"
        args = args_to_cmd(self.parameters)

        cmd = ['stdbuf', '-oL', command, '-u', script] + args

        # Start the subprocess and capture its output
        try:
            with subprocess.Popen(cmd, cwd="./Attacks/GPTFuzz", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as sp:
                for line in sp.stdout:
                    logging.info(line)
        finally:
            sp.terminate()
            sp.wait()

class DeepInception(BaseAttackModel):
    def __init__(self,model):
        # Set up default parameters
        super().__init__()
        self.parameters = {
            'target-model': model, #["vicuna", 'falcon', 'llama-2',"gpt-3.5-turbo", "gpt-4"]
            'target-max-n-tokens': 128,
            'exp_name':"main",
            'defense': 'none'
        }



    def run(self):
        print("Running main.py with specified parameters for DeepInception")
        command = "python"
        script = "main.py"
        args = args_to_cmd(self.parameters)

        cmd = ['stdbuf', '-oL', command, '-u', script] + args

        # Start the subprocess and capture its output
        try:
            with subprocess.Popen(cmd, cwd="./Attacks/DeepInception", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as sp:
                for line in sp.stdout:
                    logging.info(line)
        finally:
            sp.terminate()
            sp.wait()


class Tap(BaseAttackModel):
    def __init__(self,model):
        # Set up default parameters
        super().__init__()
        self.parameters = {
            'target-model':model, #["vicuna", 'falcon', 'llama',"gpt-3.5-turbo", "gpt-4"]
            'attack-model':'vicuna13', #gpt-3.5-turbo or local llm
            'evaluator-model':"gpt-4-1106-preview",
            'branching-factor':4,
            'width':10,
            'depth':10
        }



    def run(self):
        print("Running main.py with specified parameters for TAP")
        command = "python"
        script = "main_TAP.py"
        args = args_to_cmd(self.parameters)

        cmd = ['stdbuf', '-oL', command, '-u', script] + args

        try:
            with subprocess.Popen(cmd, cwd="./Attacks/TAP", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as sp:
                for line in sp.stdout:
                    logging.info(line)
        finally:
            sp.terminate()
            sp.wait()
            
class Pair(BaseAttackModel):
    def __init__(self,model):
        # Set up default parameters
        super().__init__()
        self.parameters = {
            'target-model':model, #["vicuna", 'falcon', 'llama-2',"gpt-3.5-turbo", "gpt-4"]
            'attack-model':'vicuna13',
            'judge-model':'gpt-4',
            'n-streams': 20,
            'keep-last-n':3,
            'n-iterations':4, ## MUST BE 4 so that the worst case generation is more than 75.
            'file-path':"../../Data/data.csv"
        }


    def run(self):
        print("Running main.py with specified parameters for PAIR")
        command = "python"
        script = "main.py"
        args = args_to_cmd(self.parameters)

        cmd = ['stdbuf', '-oL', command, '-u', script] + args

        # Start the subprocess and capture its output
        try:
            with subprocess.Popen(cmd, cwd="./Attacks/JailbreakingLLMs", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as sp:
                for line in sp.stdout:
                    logging.info(line)
        finally:
            sp.terminate()
            sp.wait()

            
class Jailbroken(BaseAttackModel):
    def __init__(self,model):
        # Set up default parameters
        super().__init__()
        self.parameters = {
            'model':model,
        }


    def run(self):
        print("Running jailbreak_all.py with specified parameters for Jailbroken")
        command = "python"
        script = "jailbreak_all.py"
        args = args_to_cmd(self.parameters)

        cmd = ['stdbuf', '-oL', command, '-u', script] + args

        # Start the subprocess and capture its output
        try:
            with subprocess.Popen(cmd, cwd="./Attacks/jailbroken/scripts/", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as sp:
                for line in sp.stdout:
                    logging.info(line)
        finally:
            sp.terminate()
            sp.wait()

class TemplateJailbreak(BaseAttackModel):
    def __init__(self,model):
        # Set up default parameters
        super().__init__()
        self.parameters = {
            'model_path':model, #[llama2,vicuna,gpt-3.5-turbo]
            'question_count':100,
        }


    def run(self):
        print("Running main.py with specified parameters for TemplateJailbreak")
        command = "python"
        script = "main.py"
        args = args_to_cmd(self.parameters)

        cmd = ['stdbuf', '-oL', command, '-u', script] + args

        # Start the subprocess and capture its output
        try:
            with subprocess.Popen(cmd, cwd="./Attacks/TemplateJailbreak/", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as sp:
                for line in sp.stdout:
                    logging.info(line)
        finally:
            sp.terminate()
            sp.wait()

class Parameters(BaseAttackModel):
    def __init__(self,model):
        # Set up default parameters
        super().__init__()
        self.parameters = {
            'model':model, #[llama2,vicuna,gpt-3.5-turbo]
            'tune_temp':"true",
            'tune_topk':"true",
            'tune_topp':"true",
            'tune_presence':"false",
            'tune_frequency':"false",            
        }


    def run(self):
        print("Running main.py with specified parameters for Parameters")
        command = "python"
        script = "attack.py"
        args = args_to_cmd(self.parameters)

        cmd = ['stdbuf', '-oL', command, '-u', script] + args

        # Start the subprocess and capture its output
        try:
            with subprocess.Popen(cmd, cwd="./Attacks/Parameter/", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as sp:
                for line in sp.stdout:
                    logging.info(line)
        finally:
            sp.terminate()
            sp.wait()





def main():
    print("The attack is starting...")
    # _init_models('vicuna13')
    # _init_models('vicuna')
    # _init_models('llama')
    # _init_models('mistral')
    # _clear_memo()

    # GCG_instance = GCG(model="vicuna")
    # GCG_instance.run()
    # AutoDan_instance = AutoDAN(model="vicuna")
    # AutoDan_instance.run()
    # AutoDan_instance = AutoDAN(model="llama")
    # AutoDan_instance.run()
    # GPTfuzz_instance = GPTFuzz(model="gpt-3.5-turbo")
    # GPTfuzz_instance.run()
    # Inception_instance = DeepInception(model= "gpt-3.5-turbo")
    # Inception_instance.run()

    # tap_instance = Tap(model="vicuna")
    # tap_instance.run()
    # tap_instance = Tap(model="llama")
    # tap_instance.run()
    
    # Pair_instance = Pair(model="vicuna")
    # Pair_instance.run()
    # Pair_instance = Pair(model="llama")
    # Pair_instance.run()
    # GCG_instance = GCG(model="vicuna")
    # GCG_instance.run()
    # GCG_instance = GCG(model="llama")
    # GCG_instance.run()
    # jailbroken_instance = Jailbroken(model="vicuna")
    # jailbroken_instance.run()
    # jailbroken_instance = Jailbroken(model="llama")
    # jailbroken_instance.run()
    # # _clear_memo()
    # templateJailbreak = TemplateJailbreak(model="vicuna")
    # templateJailbreak.run()
    # templateJailbreak = TemplateJailbreak(model="llama")
    # templateJailbreak.run()
    # # _clear_memo()
    # parameter = Parameters(model="vicuna")
    # parameter.run()
    # parameter = Parameters(model="llama")
    # parameter.run()
if __name__ == "__main__":
    main()