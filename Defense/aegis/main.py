import aegis
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
from tenacity import retry, stop_after_attempt, wait_random_exponential
import time
class AegisClass:
    def __init__(self,file_path):
        # if "gpt" in model_path:
        #     self.llm = models.OpenAILLM(model_path=model_path)
        # elif "llama" in model_path:
        #     model_path ="../../models/meta-llama/Llama-2-7b-chat-hf"
        #     # model_path = "../../llama2/llama-2-7b-chat-hf"  
        #     self.llm = models.LocalLLM(model_path=model_path)
        # elif "vicuna" in model_path:
        #     model_path ="../../models/lmsys/vicuna-7b-v1.5"  
        #     self.llm = models.LocalLLM(model_path=model_path)
        # else:
        #     raise ValueError("Unknown model name")
        
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
        api_key = os.environ.get("AEGIS_API_KEY")
        self.ag = aegis.Aegis(api_key=api_key)
        self.failed_prompts = []
        self.success_prompts_after_input = []
        self.outputs=[]
        self.fail_prompts_after_output = []
        self.success_prompts_after_output = []

        self.fail_output_after_output = []
        self.success_output_after_output = []

        self.file_path = file_path
        self.file_name = ""
        self.prompts = []

        self.system_message = "You are a helpful agent"
        self.file_name_prompts = defaultdict(list)
        self.final_results = defaultdict(list)
    def load_data(self):
        with open(self.file_path, 'r') as f:
            data = json.load(f)
        return data
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(25))
    def completion_with_backoff(self,prompt):
        flag = self.ag.ingress(self.system_message, prompt[0])

        return flag["detected"]
    
    def scan_prompt(self):
        n = 1
        for prompt in self.prompts:
            print (f"Scanning prompt {n}/{len(self.prompts)}")
            n += 1
            try:
                ingress_attack_detected = self.completion_with_backoff(prompt)
            except:
                print("Something's wrong with the aegis api, try again later")
                continue
            if ingress_attack_detected:
                print("ingress attack detected")
                self.failed_prompts.append(prompt)
            else:
                if args.input_scanner_only:
                    self.final_results[self.file_name].append(prompt)
                self.success_prompts_after_input.append(prompt)
                # if len(prompt) <3:
                #     #what is this lisr
                #     print (f"prompt {prompt} is not in the right format")
                #     self.outputs.append(prompt[2])
            # break

    # def get_outputs(self):
    #     processed_success_prompts_after_input = []
    #     for sanitized_prompt in self.success_prompts_after_input:
    #         processed_success_prompts_after_input.append(self.create_prompt(sanitized_prompt[0]))
    #     outputs = self.llm.generate_batch(processed_success_prompts_after_input)
    #     self.outputs = outputs

    def scan_output(self):
        n = 1
        for input, model_output in zip(self.success_prompts_after_input, self.outputs):
            print (f"Scanning output {n}/{len(self.success_prompts_after_input)}")
            n += 1
            try:
                 egress_attack_detected = self.ag.egress(self.system_message, model_output)["detected"]
            except:
                print("Something's wrong with the aegis api, try again later")
                sys.exit()
            if egress_attack_detected:
                print("egress attack detected")
                self.fail_prompts_after_output.append(input)
                self.fail_output_after_output.append(model_output)
            else:
                self.success_prompts_after_output.append(input)
                self.success_output_after_output.append(model_output)
                self.final_results[self.file_name].append(input)
    
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
            with open (f"../../Results/defense/{self.directory_name}/aegis_{self.model_name}_benign.json",'w') as f:
                json.dump(self.final_results,f,indent=4)
        else:
            with open (f"../../Results/defense/{self.directory_name}/aegis_{self.model_name}.json",'w') as f:
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
    parser.add_argument('--file_path', type=str, required=True, default="../../Results/gpt/Merged_gpt-3.5.json")
    parser.add_argument(
        '--benign_check',
        type=str,
        default="true"
    )
    parser.add_argument(
        '--input_scanner_only',
        type=str,
        default="true"
    )
    args = parser.parse_args()
    # model_path = args.model_path
    file_path = args.file_path
    args.benign_check = args.benign_check.lower() == "true"
    args.input_scanner_only = args.input_scanner_only.lower() == "true"
    if 'gpt' in file_path:
        name = 'gpt-3.5-turbo'
    elif 'vicuna' in file_path:
        name = 'vicuna'
    elif 'llama' in file_path:
        name = 'llama-2'
    if args.benign_check:
        file_path = f"../../Results/alpaca/alpaca_benign_{name}.json"

    ag = AegisClass(file_path)
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
            #     param,value = prompt[3][0],prompt[3][1]
            #     temp = f"{param},{value}"
            #     prompt_list.append([temp,prompt[1],prompt[2]])
            # else: 
            prompt_list.append(prompt)
        # print("attack prompts are created")
        ag.file_name_prompts[file_name] = prompt_list
    for file_name, prompts in ag.file_name_prompts.items():
        ag.file_name = file_name
        ag.prompts = prompts
        ag.scan_prompt()
        if not ag.success_prompts_after_input:
            ag.final_results[ag.file_name] = []
            ag.evaluate()
            ag.clear()
            print(f"no success prompts after input for {ag.file_name}")
            continue
        if args.input_scanner_only:
            ag.evaluate()
            ag.clear()
            continue   
        # ag.get_outputs()
        ag.scan_output()
        ag.evaluate()
        ag.clear()
    ag.save_data(start_time= start_time)



