"""Test all jailbreak prompts across all models and wrappers."""
import argparse
import logging
import os
import pathlib
import pprint
import random
from dataclasses import asdict
from uuid import uuid4
import json
import pandas as pd
__SEED = 666683321

import jinja2
import sys
# Get the directory of the current script.
current_script_path = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the current script.
parent_directory = os.path.dirname(current_script_path)

# Add the parent directory to sys.path.
sys.path.append(parent_directory)
# print(sys.path)

jinja_env = jinja2.Environment(
    loader=jinja2.PackageLoader("scripts"),
    autoescape=jinja2.select_autoescape(),
    undefined=jinja2.StrictUndefined,
)

from jailbroken.backends import ChatBackend, Request
from jailbroken.loaders import list_jailbreaks, load_backend, load_wrapper
from jailbroken.utils import init_experiment, save_datapoint_jsonl, save_experiment_json
from jailbroken.wrappers import (
    AutoObfuscationWrapper,
    DevModeWithRantiWrapper,
    MessageWrapper,
)
REPEAT_TIME_PER_QUESTION = 5

WRAPPER_CHOICES = list_jailbreaks()

PROMPT_DIR = pathlib.Path(__file__).parent / "templates/prompts"
PROMPT_CHOICES = [f.stem for f in PROMPT_DIR.glob("*.txt")]

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str,default="llama")
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--n", type=int, default=1)
parser.add_argument(
    "--system-message", type=str, default="You are a helpful assistant."
)
parser.add_argument(
    "--wrappers", choices=WRAPPER_CHOICES, default=WRAPPER_CHOICES, nargs="+"
)
parser.add_argument(
    "--prompts", choices=PROMPT_CHOICES, default=PROMPT_CHOICES, nargs="+"
)
parser.add_argument("--handle", type=str, default="jailbreak_all_v2")

# Load comment from EXPERIMENT_COMMENT environment variable
parser.add_argument("--comment", default=os.environ.get("EXPERIMENT_COMMENT", ""))


def _run(backend: ChatBackend, wrapper: MessageWrapper, prompt: str, retries: int = 3):
    """Test jailbreak prompts."""
    for i in range(retries):
        try:
            wrapper.reset()
            request = wrapper.prompt(prompt)
            temp_responses = backend.run(Request(prompt=request),from_user=True)
            response_final = []
            for response in temp_responses:
                response_final.append(wrapper.parse(response))


            result = {
                "question": prompt,
                "request": [asdict(message) for message in request],
                "response": response_final,
            }
            return result
        except Exception:
            logging.exception(f"Error running sample on attempt {i + 1}! Retrying...")
    raise RuntimeError(f"Failed to run sample after {retries} retries.")

final_result = []
def main():
    global final_result
    args = parser.parse_args()
    init_experiment(args.handle)
    save_experiment_json("args", vars(args))
    if 'vicuna' in args.model:
        model_name_file = 'vicuna'
        directory_name = 'vicuna'
        print("vicuna model is loaded")
    elif 'llama' in args.model:
        model_name_file = 'llama-2'
        directory_name='llama'
        print("llama model is loaded")
    elif 'gpt-3.5' in args.model:
        model_name_file = 'gpt-3.5'
        directory_name='gpt'
        print("gpt 3.5 model is loaded")
    elif 'gpt-4' in args.model:
        model_name_file = 'gpt-4'
        directory_name='gpt'
        print("gpt 4 model is loaded")
    else:
        model_name_file = 'unknown'
        raise ValueError("Unknown model name, supports only vicuna, llama-2, gpt-3.5 and gpt-4")

    backend, model_name = load_backend(args.model, args.temperature)

    # logging.info(
    #     f"Testing {len(args.wrappers)} jailbreaks on {len(args.prompts)} prompts with {args.n} trials each"
    # )
    jailbreak_args_list = []
    for wrapper_name in args.wrappers:
        wrapper = load_wrapper(model_name, wrapper_name, args.system_message)
        if isinstance(wrapper, AutoObfuscationWrapper):
            wrapper.set_backend(backend)
        if isinstance(wrapper, DevModeWithRantiWrapper):
            wrapper.set_backend(backend)

        # for prompt_name in args.prompts:
            # template = jinja_env.get_template(f"prompts/{prompt_name}.txt")
            # prompt = template.render().strip()
        train_data = pd.read_csv('../../../Data/data.csv')
        questions = train_data['goal'].tolist()
        for prompt in questions:
            # CURRENT_REPEAT = 0
            # while CURRENT_REPEAT < REPEAT_TIME_PER_QUESTION:
            #     CURRENT_REPEAT += 1
            jailbreak_args_list.append(
                (
                    (backend, wrapper, prompt),
                    {
                        "model": args.model,
                        "wrapper": wrapper_name,
                        "prompt": "dummy type",
                        "system_message": args.system_message,
                        # "iteration": CURRENT_REPEAT,
                    },
                )
            )
            # break
        # break

    for i in range(args.n):
        logging.info(f"Trial {i + 1} of {args.n}")

        # Shuffle jailbreak args
        random.Random(__SEED + i).shuffle(jailbreak_args_list)

        for jailbreak_args, jailbreak_str_args in jailbreak_args_list:
            # logging.info(f"Running sample:\n{pprint.pformat(jailbreak_str_args)}")
            try:
                metadata = jailbreak_str_args | {"index": i, "uuid": str(uuid4())}
                result = metadata | _run(*jailbreak_args)
                save_datapoint_jsonl("responses", result)
                i = 0
                for respose in result["response"]:
                    i+=1
                    if i >=6:
                        print(f"this shouldnt happen, since n is 5, but it did happen, so might be implementation error of vllm or jailbroken, so check it out which one is causing this error. {result['wrapper']}" )
                        break
                    final_result.append({
                        "question": result["question"],
                        "prompt": result["request"][1]["content"],
                        "response": respose,
                        "wrapper": result["wrapper"],
                        # "question_type": result["prompt"],
                        "iteration": i,

                    })


            except Exception:
                logging.exception(f"Error running sample! Skipping...")
        
    if not os.path.exists(f"../../../Results/{directory_name}"):
            os.makedirs(f"../../../Results/{directory_name}")
    with open(f"../../../Results/{directory_name}/jailbroken_{model_name_file}.json", "w") as f:
        json.dump(final_result, f, indent=4)

if __name__ == "__main__":
    main()