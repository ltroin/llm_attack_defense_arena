import pandas as pd

import json

import re
import json
from collections import defaultdict
import argparse
import sys
import os
parent_parent_dir = os.path.abspath(os.path.join(os.getcwd(), '../../'))
sys.path.append(parent_parent_dir)
# # Set the path to the 'papers' directory
# papers_path = '/workspace/papers'

# # Add the 'papers' directory to sys.path
# sys.path.append(papers_path)

# Now you can import the 'models' module
import models
import torch
import gc

# import sys

# original_sys_path = sys.path.copy()
# project_root_path = os.path.join(os.path.dirname(__file__), '../../')
# sys.path.append(project_root_path)
from global_config import get_config  
config = get_config()
# MAX_ALLOWED_ITERATION_PER_QUESTION = config.MAX_ALLOWED_ITERATION_PER_QUESTION
REPEAT_TIME_PER_QUESTION = config.REPEAT_TIME_PER_QUESTION
#a reset function to reset the sys.path
# sys.path = original_sys_path
final_res = []

def replace_names(text):
    text = re.sub(r'(?i)ChatGPT', 'Llama2', text)
    text = re.sub(r'(?i)OpenAI', 'Meta', text)
    return text
    
def process_raw_jailbreak_prompts_llama2(model_path,question_count):
    file_path = '../../Data/data.csv'
    jailbreak_path = './jailbreak-prompt.xlsx'
    model_path_dicts = {"gpt-3.5-turbo":"gpt-3.5-turbo","llama": "../../models/meta-llama/Llama-2-7b-chat-hf","vicuna" : "../../models/lmsys/vicuna-7b-v1.5"}
    model_path = model_path_dicts[model_path]
    openAI_model = False
    if 'llama' in model_path:
        model_name = 'llama-2'
        directory_name='llama'
    elif 'gpt-3.5' in model_path:
        model_name = 'gpt-3.5'
        openAI_model = True
        directory_name='gpt'
    elif 'vicuna' in model_path:
        model_name = 'vicuna'
        directory_name='vicuna'
    elif 'gpt-4' in model_path:
        model_name = 'gpt-4'
        openAI_model = True
        directory_name='gpt'
    else:
        model_name = 'unknown'
        raise ValueError("Unknown model name, supports only vicuna, llama-2, gpt-3.5 and gpt-4")
    if openAI_model:
        local_model = models.OpenAILLM(model_path)
    else:
        local_model = models.LocalVLLM(model_path)
    # df = pd.read_csv(file_path)
    # df = df.sample(frac=1).reset_index(drop=True)
    df1 = pd.read_excel(jailbreak_path)
    prompts_templates = df1['text']
    # Looping through each row
    # n = 0
    train_data = pd.read_csv(file_path)
    questions = train_data['goal'].tolist()
        
        
    for question in questions:
        # CURRENT_REPEAT = 0
        print(f"running {question}")
        # Accessing the 'goal' column for each row
        # if n == question_count:
        #     break

        # n += 1
        
        # while CURRENT_REPEAT < REPEAT_TIME_PER_QUESTION:
        #     CURRENT_REPEAT+= 1
        #     print(f"running {question} and repeat time is  {CURRENT_REPEAT}")
        prompts = []
        templates= []
        original_prompts = []
        # question_bkp = []
        counter = 0
        for template in prompts_templates:
            #skip first one, not suable
            # print(f"running {template}")
            if counter == 0:
                counter =1 
                continue
            prompt = template.replace('[INSERT PROMPT HERE]', question)
            prompt = replace_names(prompt)
            original_prompts.extend([prompt]*REPEAT_TIME_PER_QUESTION)
            prompts.append(local_model.create_conv_prompt(prompt))
            templates.extend([template]*REPEAT_TIME_PER_QUESTION)
            # question_bkp.extend([question]*REPEAT_TIME_PER_QUESTION)

        responses = local_model.generate_batch(prompts,n=REPEAT_TIME_PER_QUESTION,temperature=1)
        #flat this two dimensional list
        # responses = [item for sublist in responses for item in sublist]
        print(f"responses length are {len(responses)}")
        print(f"showing the first entry of responses {responses[0]}")
        i = 0
        for prompt,template,response in zip(original_prompts,templates,responses):
        # response = response[0].outputs[0].text
            i = i%REPEAT_TIME_PER_QUESTION + 1
            if i == 0:
                i = REPEAT_TIME_PER_QUESTION
            final_res.append({"template":template,"question":question,"prompt":prompt,"response":response,"iteration":i})
            

    # with open('processed_local_jailbreak_res_s.json', 'w') as f:
    #     json.dump(processed_local_res_s, f, indent=4)
    # with open('processed_local_jailbreak_res_f.json', 'w') as f:
    #     json.dump(processed_local_res_f, f, indent=4)

    if not os.path.exists(f"../../Results/{directory_name}"):
            os.makedirs(f"../../Results/{directory_name}")
    with open(f'../../Results/{directory_name}/TemplateJailbreak_{model_name}.json', 'w') as f:
        json.dump(final_res, f, indent=4)
    del local_model
    gc.collect()
    torch.cuda.empty_cache()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="gpt-3.5-turbo",
        help="Path to your Llama-2-7b-chat-hf directory",
    )
    parser.add_argument(
        "--question_count",
        type=int,
        default=100,
        help="how many questions you would like to test",
    )
    args = parser.parse_args()
    process_raw_jailbreak_prompts_llama2(args.model_path,args.question_count)