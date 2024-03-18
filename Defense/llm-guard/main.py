from llm_guard.input_scanners import PromptInjection,Toxicity
from llm_guard import scan_prompt
from llm_guard import scan_output
from llm_guard.output_scanners import Toxicity as ToxicityOutput
import os
import sys
parent_parent_dir = os.path.abspath(os.path.join(os.getcwd(), '../../'))
sys.path.append(parent_parent_dir)
# import models
import json
import argparse
# import prompt_process
from fastchat.model import get_conversation_template
from collections import defaultdict
import time
class LLMGuard:
    def __init__(self,file_path):

        
        if "gpt" in file_path:
            self.model_name = "gpt-3.5-turbo"
            self.directory_name = "gpt"
        elif "llama" in file_path:
            self.model_name = "llama-2"
            self.directory_name = "llama"
        elif "vicuna" in file_path:
            self.model_name = "vicuna"
            self.directory_name = "vicuna"

        else:
            self.model_name = "unknown"
            raise ValueError("Unknown model name")

        self.conv_template = get_conversation_template(
            self.model_name
        )
        if self.conv_template.name == 'llama-2':
            self.conv_template.sep2 = self.conv_template.sep2.strip()
        self.benign_check = args.benign_check
        self.failed_prompts = []
        self.success_prompts_after_input = []

        self.sanitized_prompt_list_success = []
        self.sanitized_prompt_list_fail = []

        self.outputs=[]

        self.fail_prompts_after_output = []
        self.success_prompts_after_output = []
        self.fail_output_after_output = []
        self.success_output_after_output = []

        self.file_path = file_path
        self.file_name = ""
        self.prompts = []

        self.file_name_prompts = defaultdict(list)
        self.final_results = defaultdict(list)
        self.toxicitysc, self.injectionsc=Toxicity(), PromptInjection()
        self.toxicoutput = ToxicityOutput()
    def load_data(self):
        with open(self.file_path, 'r') as f:
            data = json.load(f)
        return data

    def scan_prompts(self):
        # self.toxicitysc._threshold = 0.5
        input_scanners = [self.toxicitysc, self.injectionsc]
        n = 1
        # print(len(self.final_results[self.file_name])) 
        for prompt in self.prompts:
            print (f"Scanning prompt {n}/{len(self.prompts)}")
            n += 1
            sanitized_prompt, results_valid, results_score = scan_prompt(input_scanners, prompt[0])
            if not results_valid["Toxicity"]:
                self.failed_prompts.append(prompt)
                self.sanitized_prompt_list_fail.append([sanitized_prompt,prompt[1]])
            elif not results_valid["PromptInjection"]:
                self.failed_prompts.append(prompt)
                self.sanitized_prompt_list_fail.append([sanitized_prompt,prompt[1]])
            else:
                # if self.benign_check:
                #     self.success_prompts_after_input.append([prompt[0],])
                # else:
                self.success_prompts_after_input.append(prompt)
                # self.sanitized_prompt_list_success.append([sanitized_prompt,prompt[1]])
            # break

    # def get_outputs(self):
    #     if "alpaca" in self.file_name:
    #         print("alpaca is in the file name!")
    #         processed_sanitize_prompt_list_success = []
    #         i = 1
    #         for sanitized_prompt in self.sanitized_prompt_list_success:
    #             processed_sanitize_prompt_list_success.append(self.create_prompt(sanitized_prompt[0]))
    #         for input in processed_sanitize_prompt_list_success:
    #             print("Current generation is ", i)
    #             i+=1
    #             if input == "": raise ValueError("Empty input")
    #             output = self.llm.generate(input,temperature=0.1,max_tokens=30)

    #             self.outputs.append(output)
        # else:
        #     processed_sanitize_prompt_list_success = []
        #     for sanitized_prompt in self.sanitized_prompt_list_success:
        #         processed_sanitize_prompt_list_success.append(self.create_prompt(sanitized_prompt[0]))
        #     outputs = self.llm.generate_batch(processed_sanitize_prompt_list_success)
        #     self.outputs = outputs

    def scan_outputs(self):
        # self.toxicitysc._threshold = 0.7
        output_scanners = [self.toxicoutput]
        n =1
        # for sanitized_prompt, response_text in zip(self.sanitized_prompt_list_success, self.outputs):
        for prompt in self.success_prompts_after_input:
            print (f"Scanning output {n}/{len(self.success_prompts_after_input)}")
            n += 1
            if self.benign_check:
                response = prompt[1]
            else:
                response = prompt[2]
            if response.strip() == "":
                self.fail_prompts_after_output.append(prompt)
                # self.fail_output_after_output.append(response_text)
                continue
            # if self.benign_check:
            #     sanitized_response_text, results_valid, results_score = scan_output(
            #     output_scanners, prompt[0], response)
            # else:

            sanitized_response_text, results_valid, results_score = scan_output(
            output_scanners, prompt[0], response)
            
            if not results_valid["Toxicity"]:
                # self.fail_prompts_after_output.append(sanitized_prompt)
                self.fail_output_after_output.append(sanitized_response_text)
            else:
                # self.success_prompts_after_output.append(sanitized_prompt)
                self.success_output_after_output.append(sanitized_response_text)
                self.final_results[self.file_name].append(prompt)
    
    def evaluate(self):
        if self.benign_check:
            # print(f"{self.file_name} has {(len(self.success_prompts_after_input)/len(self.prompts))} benign prompts after input")
            # print(f"{self.file_name} has {len(self.success_prompts_after_output)/len(self.prompts)} benign prompts after output")
            benign_question_set = set(pair[0] for pair in self.final_results[self.file_name])
            total_question_set = set(pair[0] for pair in self.prompts)
            self.final_results[self.file_name].append(f"The benign question pass rate of is {len(benign_question_set)}/{len(total_question_set)}")

        else:
            # print(f"{self.file_name} has {len(self.failed_prompts)/len(self.prompts)} failed jailbreak prompts")
            # print(f"{self.file_name} has {len(self.success_prompts_after_input)/len(self.prompts)} success jailbreak prompts after input")
            # print(f"{self.file_name} has {len(self.fail_prompts_after_output)/len(self.prompts)} failed jailbreak prompts after output")
            # print(f"{self.file_name} has {len(self.success_prompts_after_output)/len(self.prompts)} success jailbreak prompts after output")
            malicious = set(pair[1] for pair in self.final_results[self.file_name])
            total_question_set = set(pair[1] for pair in self.prompts)
            # malicious_prompts = set(pair[0] for pair in self.final_results[self.file_name])
            # total_prompts = set(pair[0] for pair in self.prompts)
            self.final_results[self.file_name].append(f"The Malicious prompt pass rate is {len(self.final_results[self.file_name])}/{len(self.prompts)}")
            self.final_results[self.file_name].append(f"The Malicious question pass rate is {len(malicious)}/{len(total_question_set)}")
 
    def save_data(self,start_time):
        end_time = time.time()
        elapsed_minutes = (end_time - start_time) / 60
        self.final_results[self.file_name].append(f"Elapsed time: {elapsed_minutes} minutes")
        if not os.path.exists(f"../../Results/defense/{self.directory_name}"):
            os.makedirs(f"../../Results/defense/{self.directory_name}")
        if self.benign_check:
            with open (f"../../Results/defense/{self.directory_name}/llm_guard_{self.model_name}_benign.json",'w') as f:
                json.dump(self.final_results,f,indent=4)
        else:
            with open (f"../../Results/defense/{self.directory_name}/llm_guard_{self.model_name}.json",'w') as f:
                json.dump(self.final_results,f,indent=4)

    def create_prompt(self, prompt):

        conv_template = self.conv_template
        conv_template.append_message(conv_template.roles[0], prompt)
        conv_template.append_message(conv_template.roles[1], None)
        if 'gpt' in self.model_name:
            full_prompt = conv_template.to_openai_api_messages()
        else:
            full_prompt = conv_template.get_prompt()

        # Clear the conv template
        self.conv_template.messages = []

        return full_prompt

    def clear(self):
        self.failed_prompts = []
        self.success_prompts_after_input = []

        self.sanitized_prompt_list_success = []
        self.sanitized_prompt_list_fail = []

        self.outputs=[]

        self.fail_prompts_after_output = []
        self.success_prompts_after_output = []
        self.fail_output_after_output = []
        self.success_output_after_output = []

        self.file_path = ""
        self.file_name = ""
        self.prompts = []
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_path', type=str, required=True, default="../../llama2/llama-2-7b-chat-hf")
    parser.add_argument('--file_path', type=str, required=True, default="../../Results/Merged_llama-2.json")
    parser.add_argument(
        '--benign_check',
        type=str,
        default="false"
    )
    args = parser.parse_args()
    # model_path = args.model_path
    file_path = args.file_path
    args.benign_check = args.benign_check.lower() == "true"
    if 'gpt' in file_path:
        name = 'gpt-3.5-turbo'
    elif 'vicuna' in file_path:
        name = 'vicuna'
    elif 'llama' in file_path:
        name = 'llama-2'
    if args.benign_check:
        file_path = f"../../Results/alpaca/alpaca_benign_{name}.json"

    llm_guard = LLMGuard(file_path)
    start_time = time.time()
    with open(file_path, 'r') as f:
        data = json.load(f)
    for each in data:
        file_name = each['attack']
        prompts = each['prompts']
        prompt_list = []
        for prompt in prompts:
            # if args.benign_check:
            #     prompt_list.append([prompt,"placeholder"])
            #     continue
            # if file_name=="Parameters":
            #     #only append 1,2index
            #     prompt_list.append([prompt[1],prompt[2]])
            # else: 
            prompt_list.append(prompt)
        # print("attack prompts are created")
        llm_guard.file_name_prompts[file_name] = prompt_list
    for file_name, prompts in llm_guard.file_name_prompts.items():
        llm_guard.file_name = file_name
        llm_guard.prompts = prompts
        llm_guard.scan_prompts()
        if not llm_guard.success_prompts_after_input:
            llm_guard.final_results[llm_guard.file_name] = []
            print(f"no success prompts after input for {llm_guard.file_name}")
            llm_guard.evaluate()
            llm_guard.clear()
            continue
        # llm_guard.get_outputs()
        llm_guard.scan_outputs()
        llm_guard.evaluate()
        llm_guard.clear()
    llm_guard.save_data(start_time=start_time)



