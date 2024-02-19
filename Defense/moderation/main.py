import os
import sys
parent_parent_dir = os.path.abspath(os.path.join(os.getcwd(), '../../'))
sys.path.append(parent_parent_dir)
import openai
import json
import argparse
import time

from collections import defaultdict

class ModerationAPI:
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
            self.directory_name='benign'
            self.model_name = "benign"
            
        self.client = openai.Client(api_key=os.environ["OPENAI_API_KEY"])
        self.benign_check = args.benign_check
        self.failed_prompts = []

        self.outputs=[]

        self.file_path = file_path
        self.file_name = ""
        self.prompts = []

        self.file_name_prompts = defaultdict(list)
        self.final_results = defaultdict(list)


    def scan_outputs(self):
        # self.toxicitysc._threshold = 0.7

        i = 1
        # 0 is the question (alpaca case), or the prompt, 1 is placeholder(alpaca case), or the question, 2 is the response
        temps = []
        for prompt in self.prompts:
            print("Current input check is ", i)
            i+=1
            # if i == 3:
            #     break
            # if response.strip() == "":
            #     print("empty message generated, skipping")
            #     continue
            response = self.client.moderations.create(input=prompt[0])
            flagged = response.results[0].flagged
            if flagged:
                self.failed_prompts.append(prompt)
            else:
                temps.append(prompt)
        if not temps:
            self.final_results[self.file_name] = []
        else:
            for prompt in temps:
                if self.benign_check:
                    response = self.client.moderations.create(input=prompt[1])
                else:
                    response = self.client.moderations.create(input=prompt[2])
                flagged = response.results[0].flagged
                if flagged:
                    self.failed_prompts.append(prompt)
                else:
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
            with open (f"../../Results/defense/{self.directory_name}/moderation_{self.model_name}_benign.json",'w') as f:
                json.dump(self.final_results,f,indent=4)
        else:
            with open (f"../../Results/defense/{self.directory_name}/moderation_{self.model_name}.json",'w') as f:
                json.dump(self.final_results,f,indent=4)


    def clear(self):
        self.failed_prompts = []

        self.outputs=[]
        self.file_name = ""
        self.prompts = []

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, required=True, default="../../Results/Merged_llama-2_manual.json")
    parser.add_argument(
        '--benign_check',
        type=str,
        default="false"
    )
    args = parser.parse_args()
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

    moderation_obj = ModerationAPI(file_path)
    start_time = time.time()
    with open(file_path, 'r') as f:
        data = json.load(f)
    for each in data:
        file_name = each['attack']
        prompts = each['prompts']
        prompt_list = []
        for prompt in prompts:
            # if args.benign_check:
            #     prompt_list.append([prompt,"placeholder","placeholder"])
            #     continue
            # if file_name=="Parameters":
            #     #only append 1,2index
            #     prompt_list.append([prompt[1],prompt[2]])
            # else: 
            prompt_list.append(prompt)
        # print("attack prompts are created")
        moderation_obj.file_name_prompts[file_name] = prompt_list
    for file_name, prompts in moderation_obj.file_name_prompts.items():
        moderation_obj.file_name = file_name
        moderation_obj.prompts = prompts
        # moderation_obj.get_outputs()
        moderation_obj.scan_outputs()
        moderation_obj.evaluate()
        moderation_obj.clear()
    moderation_obj.save_data(start_time = start_time)



