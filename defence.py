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
import prompt_process

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


def args_to_cmd(args):
    cmd = []
    for key, value in args.items():
        cmd.append(f"--{key}")
        cmd.append(str(value))
    return cmd

class BaseDefenseModel:
    def __init__(self):
        self.parameters = {}

    def collect_parameters(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def Execute(self):
        raise NotImplementedError("This method should be implemented by subclasses.")


class Certified(BaseDefenseModel):
    def __init__(self,file):
        # Set up default parameters
        super().__init__()
        self.parameters = {
            'file_path': file,
            'benign_check': "false",
        }

    def run(self):
        print(f"Running certified-llm-safety")
       # Define the command and arguments
        command = "python"
        script = "detect.py"
        args = args_to_cmd(self.parameters)

        # Combine the command and arguments into a single list
        cmd =  ['stdbuf', '-oL'] + [command,'-u', script] + args
        
        # Start the subprocess and capture its output
        try:
            with subprocess.Popen(cmd, cwd="./Defense/certified-llm-safety", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,text=True,bufsize=1) as sp:
                for line in sp.stdout:
                    logging.info(line)
        finally:
            sp.terminate()
            sp.wait()


class RALLM(BaseDefenseModel):
    def __init__(self,model,file):
        # Set up default parameters
        super().__init__()
        self.parameters = {
            'model_path': model, #these two should be the same
            'file_path': file,
            'benign_check': "false",
            'sampling_num': "20",
            'max_new_tokens': "10",
            'dropping_rate': "0.3",
            'threshold': "0.2",
        }


    def run(self):
        print(f"Running RALLM")
       # Define the command and arguments
        command = "python"
        script = "main.py"
        args = args_to_cmd(self.parameters)

        # Combine the command and arguments into a single list
        cmd =  ['stdbuf', '-oL'] + [command,'-u', script] + args
        
        # Start the subprocess and capture its output
        try:
            with subprocess.Popen(cmd, cwd="./Defense/llm_defends", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,text=True,bufsize=1) as sp:
                for line in sp.stdout:
                    logging.info(line)
        finally:
            sp.terminate()
            sp.wait()



class Baseline(BaseDefenseModel):
    def __init__(self,model,file):
        # Set up default parameters
        super().__init__()
        self.parameters = {
            'model_path': model, #[vicuna,llama2',llama]
            'file_path': file, # this one and the one above should be the same
            'benign_check': "false", #test on benign data or not
            'defense_method': "paraphrase", #perplexity or paraphrase
            'perplexity_threshold': "5",
            'perplexity_window': "10",
        }

    def run(self):
        print(f"Running Baseline")
       # Define the command and arguments
        command = "python"
        script = "main.py"
        args = args_to_cmd(self.parameters)

        # Combine the command and arguments into a single list
        cmd =  ['stdbuf', '-oL'] + [command,'-u', script] + args
        
        # Start the subprocess and capture its output
        try:
            with subprocess.Popen(cmd, cwd="./Defense/baseline-defenses", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,text=True,bufsize=1) as sp:
                for line in sp.stdout:
                    logging.info(line)
        finally:
            sp.terminate()
            sp.wait()


class Aegis(BaseDefenseModel):
    def __init__(self,file):
        # Set up default parameters
        super().__init__()
        self.parameters = {
            'file_path': file,
            'input_scanner_only': "true",
            'benign_check': "false",
        }

    def run(self):
        print(f"Running aegis")
       # Define the command and arguments
        command = "python"
        script = "main.py"
        args = args_to_cmd(self.parameters)

        # Combine the command and arguments into a single list
        cmd =  ['stdbuf', '-oL'] + [command,'-u', script] + args
        
        # Start the subprocess and capture its output
        try:
            with subprocess.Popen(cmd, cwd="./Defense/aegis", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,text=True,bufsize=1) as sp:
                for line in sp.stdout:
                    logging.info(line)
        finally:
            sp.terminate()
            sp.wait()

class LLMGuard(BaseDefenseModel):
    def __init__(self,file):
        # Set up default parameters
        super().__init__()
        self.parameters = {
            # 'model_path': model, #[vicuna,llama2',llama]../../Results/Merged_llama-2-GPTfuzz.json # must be the same
            'file_path': file,
            # 'file_path': "../../Results/Merged_llama-2-GPTfuzz.json",
            'benign_check': "false",
        }

    def run(self):
        print(f"Running LLM Guard")
       # Define the command and arguments
        command = "python"
        script = "main.py"
        args = args_to_cmd(self.parameters)

        # Combine the command and arguments into a single list
        cmd =  ['stdbuf', '-oL'] + [command,'-u', script] + args
        
        # Start the subprocess and capture its output
        try:
            with subprocess.Popen(cmd, cwd="./Defense/llm-guard", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,text=True,bufsize=1) as sp:
                for line in sp.stdout:
                    logging.info(line)
        finally:
            sp.terminate()
            sp.wait()

class Smooth(BaseDefenseModel):
    def __init__(self,model,file):
        # Set up default parameters
        super().__init__()
        self.parameters = {
            'target_model': model, #[vicuna,llama2',llama]
            'attack_logfile': file,
            'smoothllm_num_copies': "8",
            'smoothllm_pert_pct': "10",
            'smoothllm_pert_type': "RandomSwapPerturbation",
            'benign_check': "false",
        }


    def run(self):
        print(f"Running Smooth")
       # Define the command and arguments
        command = "python"
        script = "main.py"
        args = args_to_cmd(self.parameters)

        # Combine the command and arguments into a single list
        cmd =  ['stdbuf', '-oL'] + [command,'-u', script] + args
        
        # Start the subprocess and capture its output
        try:
            with subprocess.Popen(cmd, cwd="./Defense/smooth-llm", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,text=True,bufsize=1) as sp:
                for line in sp.stdout:
                    logging.info(line)
        finally:
            sp.terminate()
            sp.wait()

# class Rain(BaseDefenseModel):
#     def __init__(self):
#         # Set up default parameters
#         super().__init__()
#         self.parameters = {
#             'outdir': 'test'
#         }


#     def run(self):
#         print(f"Running Smooth")
#        # Define the command and arguments
#         command = "python"
#         script = "main.py"
#         args = args_to_cmd(self.parameters)

#         # Combine the command and arguments into a single list
#         cmd =  ['stdbuf', '-oL'] + [command,'-u', script] + args
        
#         # Start the subprocess and capture its output
#         try:
#             with subprocess.Popen(cmd, cwd="./Defense/RAIN/adv", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,text=True,bufsize=1) as sp:
#                 for line in sp.stdout:
#                     logging.info(line)
#         finally:
#             sp.terminate()
#             sp.wait()

class Moderation(BaseDefenseModel):
    def __init__(self,file_path):
        # Set up default parameters
        super().__init__()
        self.parameters = {
            'file_path': file_path,
            'benign_check': "false",
        }


    def run(self):
        print(f"Running Moderation")
       # Define the command and arguments
        command = "python"
        script = "main.py"
        args = args_to_cmd(self.parameters)

        # Combine the command and arguments into a single list
        cmd =  ['stdbuf', '-oL'] + [command,'-u', script] + args
        
        # Start the subprocess and capture its output
        try:
            with subprocess.Popen(cmd, cwd="./Defense/moderation", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,text=True,bufsize=1) as sp:
                for line in sp.stdout:
                    logging.info(line)
        finally:
            sp.terminate()
            sp.wait()

class BergeonMethod(BaseDefenseModel):
    def __init__(self,file_path):
        # Set up default parameters
        super().__init__()
        self.parameters = {
            'file_path': file_path,
            'benign_check': "false",
        }


    def run(self):
        print(f"Running BergeonMethod")
       # Define the command and arguments
        command = "python"
        script = "sandbox.py"
        args = args_to_cmd(self.parameters)

        # Combine the command and arguments into a single list
        cmd =  ['stdbuf', '-oL'] + [command,'-u', script] + args
        
        # Start the subprocess and capture its output
        try:
            with subprocess.Popen(cmd, cwd="./Defense/bergeon", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,text=True,bufsize=1) as sp:
                for line in sp.stdout:
                    logging.info(line)
        finally:
            sp.terminate()
            sp.wait()


def main():
    print("The defense is starting...")
    # rallm = RALLM(model='vicuna',file="../../Results/vicuna/Merged_vicuna.json")
    # rallm.run()
    # rallm = RALLM(model='llama',file="../../Results/llama/Merged_llama-2.json")
    # rallm.run()
    # rallm = RALLM(model='gpt-3.5-turbo',file="../../Results/gpt/Merged_gpt-3.5.json")
    # rallm.run()
    # _clear_memo()
    baseline = Baseline(model='vicuna',file="../../Results/vicuna/Merged_vicuna.json")
    baseline.run()
    baseline = Baseline(model='llama',file="../../Results/llama/Merged_llama-2.json")
    baseline.run()
    baseline = Baseline(model='gpt-3.5-turbo',file="../../Results/llama/Merged_llama-2.json")
    baseline.run()

    # _clear_memo()
    # aegis = Aegis(file="../../Results/vicuna/Merged_vicuna.json")
    # aegis.run()
    # aegis = Aegis(file="../../Results/llama/Merged_llama-2.json")
    # aegis.run()
    # aegis = Aegis(file="../../Results/gpt/Merged_gpt-3.5.json")
    # aegis.run()
    # _clear_memo()
    # rebuff = Rebuff()
    # rebuff.run()
    # llm_guard = LLMGuard(model='vicuna',file="../../Results/vicuna/Merged_vicuna.json")
    # llm_guard.run()
    # llm_guard = LLMGuard(model='llama',file="../../Results/llama/Merged_llama-2.json")
    # llm_guard.run()
    # llm_guard = LLMGuard(model='gpt-3.5-turbo',file="../../Results/gpt/Merged_gpt-3.5.json")
    # llm_guard.run()
    # _clear_memo()
    # smooth = Smooth(model='vicuna',file="../../Results/vicuna/Merged_vicuna.json")
    # smooth.run()
    # smooth = Smooth(model='llama',file="../../Results/llama/Merged_llama-2.json")
    # smooth.run()
    # smooth = Smooth(model='gpt-3.5-turbo',file="../../Results/gpt/Merged_gpt-3.5.json")
    # smooth.run()
    # _clear_memo()


    # _clear_memo()
    # Rain().run()
    # moderation = Moderation(file_path="../../Results/gpt/Merged_gpt-3.5.json")
    # moderation.run()
    # # _clear_memo()
    # moderation = Moderation(file_path="../../Results/llama/Merged_llama-2.json")
    # moderation.run()
    # # _clear_memo()
    # moderation = Moderation(file_path="../../Results/vicuna/Merged_vicuna.json")
    # moderation.run()
    # _clear_memo()
    # bg = BergeonMethod(file_path="../../Results/llama/Merged_llama-2.json")
    # bg.run()
    # # _clear_memo()
    # bg = BergeonMethod(file_path="../../Results/vicuna/Merged_vicuna.json")
    # bg.run()
    # _clear_memo()
    # bg = BergeonMethod(file_path="../../Results/gpt/Merged_gpt-3.5.json")
    # bg.run()
    # _clear_memo()
    
    

    ...

if __name__ == "__main__":
    # prompt_process.load_json()
    main()