import argparse
import os
import sys
parent_parent_dir = os.path.abspath(os.path.join(os.getcwd(), '../../'))
sys.path.append(parent_parent_dir)
import json
import models
import time
from collections import defaultdict
from fastchat.model import get_conversation_template
Step 0:
##TODO import necessary libraries and define the class
"""
import xxx
"""

class CustomizedDefense:
    def __init__(self,file_path,benign_check):
        Step 3:
        ##TODO Case 1: If the this method needs the model to generate new outputs
        if "gpt" in file_path:
            self.model_name = "gpt-3.5-turbo"
            self.directory_name = "gpt"
            self.model_path = "gpt-3.5-turbo"
            self.model = models.OpenAILLM(model_path="gpt-3.5-turbo-1106")
        elif "llama" in file_path:
            self.model_name = "llama-2"
            self.directory_name = "llama"
            self.model_path = "../../models/meta-llama/Llama-2-7b-chat-hf"
            self.model = models.LocalVLLM(model_path=self.model_path)
        elif "vicuna" in file_path:
            self.model_name = "vicuna"
            self.directory_name = "vicuna"
            self.model_path = "../../models/lmsys/vicuna-7b-v1.5"
            self.model = models.LocalVLLM(model_path=self.model_path)
        else:
            self.model_name = "unknown"
            raise ValueError("Unknown model name")
        
    ##TODO Case 2: If the this method does Not need the model to generate new outputs
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
        self.benign_check = benign_check

        self.failed_prompts_after_checking_input = defaultdict(list)
        self.success_prompts_after_checking_input = []

        self.file_path = file_path
        self.file_name = ""
        self.prompts = []

        #store the results of scan_prompt
        self.input_scanning_result = defaultdict(list) 
        
        #intial dictionary to store the prompt object, such as [prompt,question,response], for each attack method
        self.file_name_prompts = defaultdict(list) 

        #The final results of the defense, containing the object passed the defense
        #when calling self.evaluate(), the pass rate will be calculated and stored in this dictionary
        self.final_results = defaultdict(list)
    
    Step 4:
    def scan_prompt(self):
        n = 1
        #For malicious input, each prompt object is a list of several elements
        # [prompt, question,response], the prompt in this case is the input to the model
        #For benign input, each prompt object is a list of two elements
        # [question, response]
        for prompt in self.prompts:
            print (f"Scanning prompt {n}/{len(self.prompts)}")
            n += 1
            if self.file_name == "Parameters":
                #if the Attack file name is parameters, then the prompt[3] stores the parameters, i.e. top_P, temperature, etc.
                params = prompt[3]
                Step 5:
                ##TODO implement your own input_processing_method!
                response = self.input_processing_method(prompt,params)
            else:
                response = self.input_processing_method(prompt)
            """
            Some examples:
            IMPORTANT: Malicious prompt object= [prompt, question,response, (parameters, in case additional element is needed)]
            Benign prompt object = [question, response]

            

            
            1. Your input_processing_method() generates a new response, then you need to update the prompt list with this "new response"
            e.g.
            if self.benign_check:
                self.input_scanning_result[self.file_name].append([prompt[0],response])
            else:
                prompt[2] = response
                self.input_scanning_result[self.file_name].append(prompt)
            
            2. Your input_processing_method() does not generate a new response, but you need to check the modifed prompt,
            if you have a customized method to check your input, e.g.

            if self.check_alignment(response):
                self.success_prompts_after_checking_input.append(prompt)
            else:
                # not align, detected by your input checking method

                self.failed_prompts_after_checking_input[self.file_name].append(prompt)
             
            
            """
            # If the input_scanning() is sufficient, you can just append it to the final_results
            # Then, comment the scan_outputs() function in the main() function
            self.final_results[self.file_name].append(prompt)
            ...

    Step 6:
    ##TODO OPTIONAL, in case you would like to check the output of the model, or the object after the scan_prompt() function
    def scan_outputs(self):
        i = 1
        for prompt in self.success_prompts_after_checking_input:
            print("Current input check is ", i)
            i+=1
            Step 7:
            ##TODO implement your own output_processing_method!
            response = self.output_processing_method(prompt)
            """
            Similar to the scan_prompt() function, you can modify the prompt object here
            In case of malicious prompt object
            prompt = [prompt[0], prompt[1],response]
            or
            prompt[2] = response

            In case of benign prompt object
            prompt[1] = response

            """
            #the prompt object passed the defense!
            if self.self.check_alignment(response):
                self.final_results[self.file_name].append(prompt)



    
    def evaluate(self):
        if self.benign_check:
            #pair[0] is the question for benign 
            benign_question_set = set(pair[0] for pair in self.final_results[self.file_name])
            total_question_set = set(pair[0] for pair in self.prompts)
            self.final_results[self.file_name].append(f"The benign question pass rate of is {len(benign_question_set)}/{len(total_question_set)}")

        else:
            #pair[1] is the question for malicious
            malicious = set(pair[1] for pair in self.final_results[self.file_name])
            total_question_set = set(pair[1] for pair in self.prompts)
            self.final_results[self.file_name].append(f"The Malicious prompt pass rate is {len(self.final_results[self.file_name])}/{len(self.prompts)}")
            self.final_results[self.file_name].append(f"The Malicious question pass rate is {len(malicious)}/{len(total_question_set)}")
    
    def save_data(self,start_time):
        end_time = time.time()
        elapsed_minutes = (end_time - start_time) / 60
        self.final_results[self.file_name].append(f"Elapsed time: {elapsed_minutes} minutes")
        if not os.path.exists(f"../../Results/defense/{self.directory_name}"):
            os.makedirs(f"../../Results/defense/{self.directory_name}")
        if self.benign_check:
            with open (f"../../Results/defense/{self.directory_name}/{YOUR_DEFENSE_NAME}_{self.model_name}_benign.json",'w') as f:
                json.dump(self.final_results,f,indent=4)
        else:
            with open (f"../../Results/defense/{self.directory_name}/{YOUR_DEFENSE_NAME}_{self.model_name}.json",'w') as f:
                json.dump(self.final_results,f,indent=4)


    def clear(self):

        self.failed_prompts_after_checking_input = []
        self.success_prompts_after_checking_input = []

        self.file_name = ""
        self.prompts = []
        




def main():
    Step 1:
    ##TODO modify the entry with args necessary for your method
    parser = argparse.ArgumentParser()
    #add the necessary arguments
    # parser.add_argument('--file_path', type=str, default="../../Results/gpt/Merged_gpt-3.5.json")
    args = parser.parse_args()
    file_path = args.file_path
    #only used for the benign check
    args.benign_check = args.benign_check.lower() == "true"
    
    if 'gpt' in file_path:
        name = 'gpt-3.5-turbo'
    elif 'vicuna' in file_path:
        name = 'vicuna'
    elif 'llama' in file_path:
        name = 'llama-2'

    if args.benign_check:
        file_path = f"../../Results/alpaca/alpaca_benign_{name}.json"

        
    customized_defense =  CustomizedDefense(file_path=file_path,benign_check=args.benign_check)
    start_time = time.time()
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    for each in data:
        file_name = each['attack']
        prompts = each['prompts']
        prompt_list = []
        for prompt in prompts:
            prompt_list.append(prompt)
        customized_defense.file_name_prompts[file_name] = prompt_list
    
    Step 2:
    ##TODO Decide whether to scan the input and also the output   
    # Structure: AttackName, [prompt,question,response] or [question,response] in case of benign check
    
    for file_name, prompts in customized_defense.file_name_prompts.items():
        customized_defense.file_name = file_name
        customized_defense.prompts = prompts
        customized_defense.scan_prompt()
        ##In case all prompts are filtered out by the scan_prompt method, which checks the input.
        if not customized_defense.success_prompts_after_checking_input:
            customized_defense.final_results[customized_defense.file_name] = []
            customized_defense.evaluate()
            customized_defense.clear()
            print(f"no success prompts after input for {customized_defense.file_name}")
            continue

        ##TODO OPTIONAL, in case you would like to check the output of the model
        customized_defense.scan_outputs()

        customized_defense.evaluate()
        customized_defense.clear()

    
    customized_defense.save_data(start_time= start_time)


if __name__ == '__main__':
    main()