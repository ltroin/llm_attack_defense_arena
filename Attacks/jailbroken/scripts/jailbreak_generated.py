"""Test all jailbreak prompts across all models and wrappers."""
import argparse
import json
import logging
import os
import pathlib
import pprint
import random
from dataclasses import asdict
from uuid import uuid4

__SEED = 666683321

import jinja2

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

WRAPPER_CHOICES = list_jailbreaks()


parser = argparse.ArgumentParser()
parser.add_argument("model", type=str)
parser.add_argument("prompt_file", type=str)
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--n", type=int, default=1)
parser.add_argument(
    "--system-message", type=str, default="You are a helpful assistant."
)
parser.add_argument(
    "--wrappers", choices=WRAPPER_CHOICES, default=WRAPPER_CHOICES, nargs="+"
)
parser.add_argument("--handle", type=str, default="jailbreak_generated")

# Load comment from EXPERIMENT_COMMENT environment variable
parser.add_argument("--comment", default=os.environ.get("EXPERIMENT_COMMENT", ""))


def _run(backend: ChatBackend, wrapper: MessageWrapper, prompt: str, retries: int = 3):
    """Test jailbreak prompts."""
    for i in range(retries):
        try:
            wrapper.reset()
            request = wrapper.prompt(prompt)
            response = wrapper.parse(backend.run(Request(prompt=request)))
            result = {
                "request": [asdict(message) for message in request],
                "response": response,
            }
            return result
        except Exception:
            logging.exception(f"Error running sample on attempt {i + 1}! Retrying...")
    raise RuntimeError(f"Failed to run sample after {retries} retries.")


def main():
    args = parser.parse_args()
    init_experiment(args.handle)
    save_experiment_json("args", vars(args))

    # Load prompts
    with open(args.prompt_file, "r") as f:
        prompts = json.load(f)

    backend, model_name = load_backend(args.model, args.temperature)

    logging.info(
        f"Testing {len(args.wrappers)} jailbreaks on {len(prompts)} prompts with {args.n} trials each"
    )
    jailbreak_args_list = []
    for wrapper_name in args.wrappers:
        wrapper = load_wrapper(model_name, wrapper_name, args.system_message)
        if isinstance(wrapper, AutoObfuscationWrapper):
            wrapper.set_backend(backend)
        if isinstance(wrapper, DevModeWithRantiWrapper):
            wrapper.set_backend(backend)

        for prompt_name in prompts:
            prompt = prompts[prompt_name]
            jailbreak_args_list.append(
                (
                    (backend, wrapper, prompt),
                    {
                        "model": args.model,
                        "wrapper": wrapper_name,
                        "prompt": prompt_name,
                        "system_message": args.system_message,
                    },
                )
            )

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
            except Exception:
                logging.exception(f"Error running sample! Skipping...")


if __name__ == "__main__":
    main()
