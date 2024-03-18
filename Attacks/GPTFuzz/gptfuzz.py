import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # for debugging

from fastchat.model import add_model_args
import argparse
import pandas as pd
from gptfuzzer.fuzzer.selection import MCTSExploreSelectPolicy
from gptfuzzer.fuzzer.mutator import (
    MutateRandomSinglePolicy, OpenAIMutatorCrossOver, OpenAIMutatorExpand,
    OpenAIMutatorGenerateSimilar, OpenAIMutatorRephrase, OpenAIMutatorShorten)
from gptfuzzer.fuzzer import GPTFuzzer
from gptfuzzer.llm import OpenAILLM, LocalVLLM, LocalLLM, PaLM2LLM, ClaudeLLM
from gptfuzzer.utils.predict import RoBERTaPredictor
import random
random.seed(100)
import logging
httpx_logger: logging.Logger = logging.getLogger("httpx")
# disable httpx logging
httpx_logger.setLevel(logging.WARNING)

import sys

original_sys_path = sys.path.copy()
project_root_path = os.path.join(os.path.dirname(__file__), '../../')
sys.path.append(project_root_path)
from global_config import get_config  
config = get_config()
MAX_ALLOWED_ITERATION_PER_QUESTION = config.MAX_ALLOWED_ITERATION_PER_QUESTION
REPEAT_TIME_PER_QUESTION = config.REPEAT_TIME_PER_QUESTION
#a reset function to reset the sys.path
sys.path = original_sys_path

def main(args):
    model_name = ""
    if "llama" in args.target_model:
        model_name = "llama-2"
        directory_name = "llama"
        print("llama model is loaded")
    elif "vicuna" in args.target_model:
        model_name = "vicuna"
        directory_name = "vicuna"
        print("vicuna model is loaded")
    elif "gpt-3.5" in args.target_model:
        model_name = "gpt-3.5"
        directory_name = "gpt"
    elif "gpt-4" in args.target_model:
        model_name = "gpt-4"
        directory_name = "gpt"
    else:
        model_name = "unknown"
    initial_seed = pd.read_csv(args.seed_path)['text'].tolist()
    openai_key = os.getenv('OPENAI_API_KEY')
    openai_model = OpenAILLM(args.model_path, openai_key)
    # target_model = PaLM2LLM(args.target_model, args.palm_key)
    # target_model = ClaudeLLM(args.target_model, args.claude_key)
    # target_model = LocalVLLM(args.target_model)
    model_path_dicts = {"llama-2": "../../models/meta-llama/Llama-2-7b-chat-hf","vicuna" : "../../models/lmsys/vicuna-7b-v1.5"}
    if 'gpt' not in args.target_model:
        path = model_path_dicts[model_name]
    target_model = None
    if 'gpt' in args.target_model:
        target_model = OpenAILLM(args.target_model)
    else:
        target_model = LocalVLLM(path) # we suggest using LocalVLLM for better performance, however if you are facing difficulties in installing vllm, you can use LocalLLM instead
    roberta_model = RoBERTaPredictor('hubert233/GPTFuzz', device='cuda:0')
    #import question_list.csv, the first is index, the second is question, only extract the second column
    # questions = ""
    # with open('../../Data/data.csv', 'r') as f:
    train_data = pd.read_csv('../../Data/data.csv')
    questions = train_data['goal'].tolist()
        # questions = [question.split(',')[0] for question in questions[1:]]
    


    fuzzer = GPTFuzzer(
        directory_name=directory_name,
        model_name=model_name,
        questions=questions,
        # target_model=openai_model,
        target=target_model,
        predictor=roberta_model,
        initial_seed=initial_seed,
        mutate_policy=MutateRandomSinglePolicy([
            OpenAIMutatorCrossOver(openai_model, temperature=1.0),  # for reproduction only, if you want better performance, use temperature>0
            OpenAIMutatorExpand(openai_model, temperature=1.0),
            OpenAIMutatorGenerateSimilar(openai_model, temperature=1.0),
            OpenAIMutatorRephrase(openai_model, temperature=1.0),
            OpenAIMutatorShorten(openai_model, temperature=1.0)],
            concatentate=True,
        ),
        select_policy=MCTSExploreSelectPolicy(),
        energy=args.energy,
        max_jailbreak=args.max_jailbreak,
        max_query=args.max_query,
        generate_in_batch=True,
        max_prompt_variety=args.max_prompt_variety,
    )

    fuzzer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fuzzing parameters')
    parser.add_argument('--openai_key', type=str, default='', help='OpenAI API Key')
    parser.add_argument('--claude_key', type=str, default='', help='Claude API Key')
    parser.add_argument('--palm_key', type=str, default='', help='PaLM2 api key')
    parser.add_argument('--model_path', type=str, default='gpt-3.5-turbo',
                        help='mutate model path')
    parser.add_argument('--target_model', type=str, default='vicuna',
                        help='The target model, openai model or open-sourced LLMs')
    parser.add_argument('--max_query', type=int, default=1000,
                        help='The maximum number of queries')
    parser.add_argument('--max_jailbreak', type=int,
                        default=-1, help='The maximum jailbreak number')
    parser.add_argument('--energy', type=int, default=1,
                        help='The energy of the fuzzing process')
    parser.add_argument('--seed_selection_strategy', type=str,
                        default='round_robin', help='The seed selection strategy')
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed_path", type=str,
                        default="./datasets/prompts/GPTFuzzer.csv")
    parser.add_argument("--max-prompt-variety", type=int, default=5)
    add_model_args(parser)

    args = parser.parse_args()
    main(args)
