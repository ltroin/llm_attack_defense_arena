import sys
import os
import argparse
import json
import models
import torch
parent_parent_dir = os.path.abspath(os.path.join(os.getcwd(), '../../'))
sys.path.append(parent_parent_dir)
#For generative methods, you can set the maximum number of iterations allowed per question
MAX_ALLOWED_ITERATION_PER_QUESTION = 75
#Repetition time for each question
REPEAT_TIME_PER_QUESTION = 5

Step 2:
##TODO import the modules required in your method
"""
import fastchat
import openai
...
"""

Step 3:
##TODO Define your own program logic
def main(args):
    model_path_dicts = {"gpt-4": "gpt-4-1106-preview","gpt-3.5-turbo":"gpt-3.5-turbo","llama": "../../models/meta-llama/Llama-2-7b-chat-hf","vicuna" : "../../models/lmsys/vicuna-7b-v1.5"}
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
        """
        The default value usage of generate method is as follows:
        output = local_model.generate(local_model.create_conv_prompt(prompt), max_tokens=100,temperature=1.0,top_p=1.0,frequency_penalty=0.0,presence_penalty=0.0,n=1)
        """
        local_model = models.OpenAILLM(model_path)
    else:
        """
        if not use LocalVLLM, you can use LocalLLM.
        To load another model into the second GPU  
        use the following code:
        local_model = models.LocalLLM2(model_path)

        The default value usage of generate method is as follows:
        output = local_model.generate(local_model.create_conv_prompt(prompt), temperature=1.0, max_tokens=150,top_p=1,top_k=-1, repetition_penalty=1.0,frequency_penalty=0.0,presence_penalty=0.0,n=1)
        """
        local_model = models.LocalVLLM(model_path)
    """
    To use the checker, you can use the following code:
    """


    finished_questions=set()
    
    Step 4:
    ##TODO load the data
    data = load_data()
    results = []
    
    Step 5:
    ##CASE 1: generative methods
    for question in data:
        CURRENT_ITERATION = 0
        CURRENT_REPEAT = 0
        while CURRENT_REPEAT < REPEAT_TIME_PER_QUESTION and CURRENT_ITERATION < MAX_ALLOWED_ITERATION_PER_QUESTION:
            CURRENT_REPEAT += 1
            for j in range(num_steps):
                if question in finished_questions:
                    raise ValueError("This question has been finished, you are not expected to see this error message. Please check the code.")
                if CURRENT_ITERATION >= MAX_ALLOWED_ITERATION_PER_QUESTION:
                        finished_questions.add(question)
                        CURRENT_REPEAT = REPEAT_TIME_PER_QUESTION
                        break
                CURRENT_ITERATION += 1
                
                Step 6:
                ##TODO Your attack logic
                
                prompt = ...
                output =...

                
                Step 7:
                ##save the results
                record = {
                            'prompt': prompt,
                            'response': output,
                            'question': question,
                            "iteration": CURRENT_ITERATION,
                            "other": "other information you want to save"
                        }
                results.append(record)
    Step 5:
    ##CASE 2: Other type methods, for example, Template-based methods
    for question in data:

        Step 6:
        ##TODO Your attack logic
        prompts = some_processing(question)
        responses = local_model.generate_batch(prompts,n=REPEAT_TIME_PER_QUESTION)
        questions = [question]*REPEAT_TIME_PER_QUESTION

        Step 7:
        ##save the results
        i = 0
        for prompt, question, response in zip(prompts,responses,questions):
        # response = response[0].outputs[0].text
            i = i%REPEAT_TIME_PER_QUESTION + 1
            if i == 0:
                i = REPEAT_TIME_PER_QUESTION
            results.append({"question":question,"prompt":prompt,"response":response,"iteration":i})


    if not os.path.exists(f"../../../Results/{directory_name}"):
            os.makedirs(f"../../../Results/{directory_name}")
    with open(f"../../../Results/{directory_name}/{Your_method_Name}_{model_name}.json", 'w') as f:
        json.dump(results, f, indent=4)

    

Step 1:
##TODO modify the entry with args necessary for your method
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    """
    parser.add_argument(
        "--model_path",
        type=str,
        default="gpt-3.5-turbo",
        help="Name of the model to be attacked",
    )
    ...
    """
    args = parser.parse_args()
    main(args)