import pandas as pd
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
from tqdm import tqdm
# from configs import modeltype2path
import warnings
import sys
import os
parent_parent_dir = os.path.abspath(os.path.join(os.getcwd(), '../../'))
sys.path.append(parent_parent_dir)
import models
logging.basicConfig(level=logging.INFO)
warnings.simplefilter("ignore")
import json

REPEAT_TIME_PER_QUESTION = 5

DEFAULT_SYSTEM_PROMPT = """<<SYS>> You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. <</SYS>> """
final_results= []

def prepend_sys_prompt(sentence, args):
    if args.use_system_prompt:
        sentence = DEFAULT_SYSTEM_PROMPT + sentence
    return sentence


def get_sentence_embedding(model, tokenizer, sentence):
    sentence = sentence.strip().replace('"', "")
    word_embeddings = model.get_input_embeddings()

    # Embed the sentence
    tokenized = tokenizer(sentence, return_tensors="pt", add_special_tokens=False).to(
        model.device
    )
    embedded = word_embeddings(tokenized.input_ids)
    return embedded


def main():
    global final_results
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, help="which model to use", default="gpt-3.5-turbo"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=1,
        help="how many results we generate for the sampling-based decoding",
    )
    parser.add_argument(
        "--use_greedy",type=str,default="false",help="enable the greedy decoding"
    )
    parser.add_argument(
        "--use_default", type=str,default="false", help="enable the default decoding"
    )
    parser.add_argument(
        "--tune_temp", type=str,default="true", help="enable the tuning of temperature"
    )
    parser.add_argument(
        "--tune_topp", type=str,default="false", help="enable the tuning of top_p"
    )
    parser.add_argument(
        "--tune_topk", type=str,default="false", help="enable the tuning of top_k"
    )
    parser.add_argument(
        "--tune_presence", type=str,default="false", help="enable the tuning of presence"
    )
    parser.add_argument(
        "--tune_frequency", type=str,default="false", help="enable the tuning of frequency"
    )

    parser.add_argument(
        "--use_system_prompt", action="store_true", help="enable the system prompt"
    )
    parser.add_argument(
        "--use_advbench",
        action="store_true",
        help="use the advbench dataset for evaluation",
    )
    args = parser.parse_args()
    args.tune_temp = args.tune_temp.lower() == "true"
    args.tune_topp = args.tune_topp.lower() == "true"
    args.tune_topk = args.tune_topk.lower() == "true"
    args.tune_presence = args.tune_presence.lower() == "true"
    args.tune_frequency = args.tune_frequency.lower() == "true"

    model_path_dicts = {"gpt-3.5-turbo":"gpt-3.5-turbo","llama": "../../models/meta-llama/Llama-2-7b-chat-hf","vicuna" : "../../models/lmsys/vicuna-7b-v1.5"}
    model_path = model_path_dicts[args.model]
    # model_name = modeltype2path[args.model]
    openAI_model = False
    WEIGHTS_PATH = model_path
    TOKENIZER_PATH = WEIGHTS_PATH
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
        model = models.OpenAILLM(model_path)
    else:
        if "falcon" in args.model or "mpt" in args.model:
            model = AutoModelForCausalLM.from_pretrained(
                WEIGHTS_PATH,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                WEIGHTS_PATH,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)


    fname = args.model
    if args.use_system_prompt:
        fname += "_with_sys_prompt"
    if args.n_sample > 1:
        fname += f"_sample_{args.n_sample}"
    if args.use_advbench:
        fname += "_advbench"
    if not os.path.exists(f"outputs/{fname}"):
        os.makedirs(f"outputs/{fname}")



    # logging.info(f"Model size: {model.get_memory_footprint()/1e9}")
    logging.info(f"Model name: {fname}")

    # if args.use_advbench:
    #     with open("./data/advbench.txt") as f:
    #         lines = f.readlines()[:100]
    # else:
    #     with open("./data/MaliciousInstruct.txt") as f:
    #         lines = f.readlines()

    train_data = pd.read_csv('../../Data/data.csv')
    lines = train_data['goal'].tolist()
    # prepend sys prompt
    # lines = [prepend_sys_prompt(l, args) for l in lines]

    # if args.use_greedy:
    #     logging.info(f"Running greedy")
    #     prompts = []
    #     outputs = []
    #     model.eval()

    #     for sentence in tqdm(lines):
    #         try:
    #             if "falcon" in args.model or "mpt" in args.model:
    #                 ground_truth_generation = model.generate(
    #                     tokenizer(sentence, return_tensors="pt").input_ids.to("cuda"),
    #                     max_new_tokens=100,
    #                     do_sample=False,
    #                     num_return_sequences=1,
    #                 )
    #             else:
    #                 # get ground truth generation
    #                 ground_truth_embeds = get_sentence_embedding(
    #                     model, tokenizer, sentence
    #                 )
    #                 ground_truth_generation = model.generate(
    #                     inputs_embeds=ground_truth_embeds,
    #                     max_new_tokens=100,
    #                     do_sample=False,
    #                     num_return_sequences=1,
    #                 )
    #             ground_truth_generation = tokenizer.batch_decode(
    #                 ground_truth_generation
    #             )
    #             outputs.extend(ground_truth_generation)
    #             prompts.extend([sentence] * args.n_sample)
    #         except:
    #             continue
    #         results = pd.DataFrame()
    #         results["prompt"] = [line.strip() for line in prompts]
    #         results["output"] = outputs
            # results.to_csv(f"outputs/{fname}/output_greedy.csv")

    # if args.use_default:
    #     logging.info(f"Running default, top_p=0.9, temp=0.1")
    #     prompts = []
    #     outputs = []
    #     model.eval()

    #     for sentence in tqdm(lines):
    #         try:
    #             if "falcon" in args.model or "mpt" in args.model:
    #                 ground_truth_generation = model.generate(
    #                     tokenizer(sentence, return_tensors="pt").input_ids.to("cuda"),
    #                     max_new_tokens=100,
    #                     do_sample=True,
    #                     top_p=0.9,
    #                     temperature=0.1,
    #                     num_return_sequences=1,
    #                 )
    #             else:
    #                 # get ground truth generation
    #                 ground_truth_embeds = get_sentence_embedding(
    #                     model, tokenizer, sentence
    #                 )
    #                 ground_truth_generation = model.generate(
    #                     inputs_embeds=ground_truth_embeds,
    #                     max_new_tokens=100,
    #                     do_sample=True,
    #                     top_p=0.9,
    #                     temperature=0.1,
    #                     num_return_sequences=1,
    #                 )
    #             ground_truth_generation = tokenizer.batch_decode(
    #                 ground_truth_generation
    #             )
    #             outputs.extend(ground_truth_generation)
    #             prompts.extend([sentence] * args.n_sample)
    #         except:
    #             continue
    #         results = pd.DataFrame()
    #         results["prompt"] = [line.strip() for line in prompts]
    #         results["output"] = outputs
            # results.to_csv(f"outputs/{fname}/output_default.csv")

    if args.tune_temp:
        for temp in np.arange(0.05, 1.05, 0.05):
            temp = np.round(temp, 2)
            logging.info(f"Running temp = {temp}")
            prompts = []
            outputs = []
            if not openAI_model:
                model.eval()
            
            for sentence in tqdm(lines):
                # CURRENT_REPEAT = 0
                # while CURRENT_REPEAT < REPEAT_TIME_PER_QUESTION:
                    try:
                        # if "falcon" in args.model or "mpt" in args.model:
                        #     ground_truth_generation = model.generate(
                        #         tokenizer(sentence, return_tensors="pt").input_ids.to(
                        #             "cuda"
                        #         ),
                        #         max_new_tokens=100,
                        #         temperature=temp,
                        #         do_sample=True,
                        #         num_return_sequences=args.n_sample,
                        #     )
                        # else:
                            # get ground truth generation
                        # CURRENT_REPEAT+= 1
                        if not openAI_model:
                            ground_truth_embeds = get_sentence_embedding(
                                model, tokenizer, sentence
                            )
                            ground_truth_generation = model.generate(
                                inputs_embeds=ground_truth_embeds,
                                max_new_tokens=100,
                                temperature=temp,
                                do_sample=True,
                                num_return_sequences=REPEAT_TIME_PER_QUESTION,
                            )
                            ground_truth_generation = tokenizer.batch_decode(
                                ground_truth_generation
                            )
                        else:
                            ground_truth_generation = model.generate(
                                model.create_conv_prompt(sentence,system_message=False), temperature=temp, max_tokens=100,n=REPEAT_TIME_PER_QUESTION
                            )
                        for i in range(REPEAT_TIME_PER_QUESTION):
                            final_results.append({"prompt":sentence,"question":sentence,"response":ground_truth_generation[i],"param":{'temperature':temp},"iteration":i+1})
                        
                        # outputs.append(ground_truth_generation)
                        # prompts.extend([sentence] * args.n_sample)
                        
                    except:
                        continue

            # results = pd.DataFrame()
            # results["prompt"] = [line.strip() for line in prompts]
            # results["output"] = outputs
            # results.to_csv(f"outputs/{fname}/output_temp_{temp}.csv")

    if args.tune_topp:
        for top_p in np.arange(0, 1.05, 0.05):
            top_p = np.round(top_p, 2)
            logging.info(f"Running topp = {top_p}")
            outputs = []
            prompts = []
            if not openAI_model:
                model.eval()

            for sentence in tqdm(lines):
                # CURRENT_REPEAT = 0
                # while CURRENT_REPEAT < REPEAT_TIME_PER_QUESTION:
                    try:
                        # get ground truth generation
                        # if "falcon" in args.model or "mpt" in args.model:
                        #     ground_truth_generation = model.generate(
                        #         tokenizer(sentence, return_tensors="pt").input_ids.to(
                        #             "cuda"
                        #         ),
                        #         max_new_tokens=100,
                        #         top_p=top_p,
                        #         do_sample=True,
                        #         num_return_sequences=args.n_sample,
                        #     )
                        # CURRENT_REPEAT+= 1 
                        if not openAI_model:
                            ground_truth_embeds = get_sentence_embedding(
                                model, tokenizer, sentence
                            )
                            ground_truth_generation = model.generate(
                                inputs_embeds=ground_truth_embeds,
                                max_new_tokens=100,
                                top_p=top_p,
                                do_sample=True,
                                num_return_sequences=REPEAT_TIME_PER_QUESTION,
                            )
                            ground_truth_generation = tokenizer.batch_decode(
                                ground_truth_generation
                            )
                        else:
                            ground_truth_generation = model.generate(
                                model.create_conv_prompt(sentence,system_message=False), top_p=top_p, max_tokens=100,n=REPEAT_TIME_PER_QUESTION
                            )
                        for i in range(REPEAT_TIME_PER_QUESTION):
                            final_results.append({"prompt":sentence,"question":sentence,"response":ground_truth_generation[i],"param":{'top_p':top_p},"iteration":i+1})
                        
                    except:
                        continue
                     
                # results = pd.DataFrame()
                # results["prompt"] = [line.strip() for line in prompts]
                # results["output"] = outputs
                # results.to_csv(f"outputs/{fname}/output_topp_{top_p}.csv")

    if args.tune_topk:
        for top_k in [1, 2, 5, 10, 20, 50, 100, 200, 500]:
            logging.info(f"Running topk = {top_k}")
            outputs = []
            prompts = []
            if not openAI_model:
                model.eval()

            for sentence in tqdm(lines):
                # CURRENT_REPEAT = 0
                # while CURRENT_REPEAT < REPEAT_TIME_PER_QUESTION:
                    try:
                        # get ground truth generation
                        # if "falcon" in args.model or "mpt" in args.model:
                        #     ground_truth_generation = model.generate(
                        #         tokenizer(sentence, return_tensors="pt").input_ids.to(
                        #             "cuda"
                        #         ),
                        #         max_new_tokens=100,
                        #         top_k=top_k,
                        #         do_sample=True,
                        #         num_return_sequences=args.n_sample,
                        #     )
                        # CURRENT_REPEAT+=1
                        ground_truth_embeds = get_sentence_embedding(
                            model, tokenizer, sentence
                        )
                        ground_truth_generation = model.generate(
                            inputs_embeds=ground_truth_embeds,
                            max_new_tokens=100,
                            top_k=top_k,
                            do_sample=True,
                            num_return_sequences=REPEAT_TIME_PER_QUESTION,
                        )
                        ground_truth_generation = tokenizer.batch_decode(
                            ground_truth_generation
                        )
                        # outputs.extend(ground_truth_generation)
                        # prompts.extend([sentence] * args.n_sample)
                        for i in range(REPEAT_TIME_PER_QUESTION):
                            final_results.append({"prompt":sentence,"question":sentence,"response":ground_truth_generation[i],"param":{'top_k':top_k},"iteration":i+1})
                       
                    except:
                        continue
                    
                # results = pd.DataFrame()
                # results["prompt"] = [line.strip() for line in prompts]
                # results["output"] = outputs
                # results.to_csv(f"outputs/{fname}/output_topk_{top_k}.csv")

    if args.tune_presence:
        for presence in np.arange(-2, 2.1, 0.1):
            presence = np.round(presence, 2)
            logging.info(f"Running presence penalty = {presence}")
            if not openAI_model:
                raise ValueError("Presence penalty is only tested for OpenAI models")
            for sentence in tqdm(lines):
                # CURRENT_REPEAT = 0
                # while CURRENT_REPEAT < REPEAT_TIME_PER_QUESTION:
                    try:
                        # get ground truth generation
                        # if "falcon" in args.model or "mpt" in args.model:
                        #     ground_truth_generation = model.generate(
                        #         tokenizer(sentence, return_tensors="pt").input_ids.to(
                        #             "cuda"
                        #         ),
                        #         max_new_tokens=100,
                        #         top_k=top_k,
                        #         do_sample=True,
                        #         num_return_sequences=args.n_sample,
                        #     )
                        # CURRENT_REPEAT+=1
                        ground_truth_generation = model.generate(
                            model.create_conv_prompt(sentence,system_message=False), presence_penalty=presence, max_tokens=100,n=REPEAT_TIME_PER_QUESTION
                        )
                        # outputs.extend(ground_truth_generation)
                        # prompts.extend([sentence] * args.n_sample)
                        for i in range(REPEAT_TIME_PER_QUESTION):
                            final_results.append({"prompt":sentence,"question":sentence,"response":ground_truth_generation[i],"param":{'presence':presence},"iteration":i+1})
 
                        
                    except:
                        continue
                    
            #     break
            # break
    if args.tune_frequency:
        for frequency in np.arange(-2, 2.1, 0.1):
            frequency = np.round(frequency, 2)
            logging.info(f"Running frequency penalty = {frequency}")
            if not openAI_model:
                raise ValueError("frequency penalty is only tested for OpenAI models")
            for sentence in tqdm(lines):
                # CURRENT_REPEAT = 0
                # while CURRENT_REPEAT < REPEAT_TIME_PER_QUESTION:
                    try:
                        # get ground truth generation
                        # if "falcon" in args.model or "mpt" in args.model:
                        #     ground_truth_generation = model.generate(
                        #         tokenizer(sentence, return_tensors="pt").input_ids.to(
                        #             "cuda"
                        #         ),
                        #         max_new_tokens=100,
                        #         top_k=top_k,
                        #         do_sample=True,
                        #         num_return_sequences=args.n_sample,
                        #     )
                        # CURRENT_REPEAT+=1
                        ground_truth_generation = model.generate(
                            model.create_conv_prompt(sentence,system_message=False), frequency_penalty=frequency, max_tokens=100,n=REPEAT_TIME_PER_QUESTION
                        )
                        # outputs.extend(ground_truth_generation)
                        # prompts.extend([sentence] * args.n_sample)
                        for i in range(REPEAT_TIME_PER_QUESTION):
                            final_results.append({"prompt":sentence,"question":sentence,"response":ground_truth_generation[i],"param":{'frequency':frequency},"iteration":i+1})
                         
                    except:
                        continue
                    
            #     break
            # break
    if not os.path.exists(f"../../Results/{directory_name}"):
            os.makedirs(f"../../Results/{directory_name}")
    with open(f'../../Results/{directory_name}/Parameters_{model_name}.json', 'w') as f:
        json.dump(final_results, f, indent=4)
if __name__ == "__main__":
    main()

