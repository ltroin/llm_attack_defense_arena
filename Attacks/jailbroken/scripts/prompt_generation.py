"""Start a chat with a language model."""

import argparse
import json
import logging
import os
import pathlib
import random

import jinja2

jinja_env = jinja2.Environment(
    loader=jinja2.PackageLoader("scripts"),
    autoescape=jinja2.select_autoescape(),
    undefined=jinja2.StrictUndefined,
)

from jailbroken.backends import Request
from jailbroken.loaders import load_backend, load_wrapper
from jailbroken.utils import init_experiment, save_experiment_json

PROMPTS_PATH_STR = str(pathlib.Path(__file__).parent / "templates/prompts")
BATCH_SIZE = 20

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--prompts_path", type=str, default=str(PROMPTS_PATH_STR))
parser.add_argument("--sample-size", type=int, default=10)
parser.add_argument(
    "--template", type=str, default="other_prompts/prompt_generation.jinja"
)
parser.add_argument(
    "--system-message",
    type=str,
    default="You are a helpful assistant that always responds in JSON format.",
)

# Load comment from EXPERIMENT_COMMENT environment variable
parser.add_argument("--comment", default=os.environ.get("EXPERIMENT_COMMENT", ""))


def dump_prompts(prompts_path: pathlib.Path):
    prompt_paths = list(prompts_path.iterdir())
    random.shuffle(prompt_paths)
    prompts = {}
    for p in prompt_paths:
        if p.suffix == ".txt":
            prompts[p.stem] = p.read_text()
    return json.dumps(prompts, indent=4)


def main():
    args = parser.parse_args()
    init_experiment("prompt_generation")
    save_experiment_json("args", vars(args))

    template = jinja_env.get_template(args.template)
    prompts_json = dump_prompts(pathlib.Path(args.prompts_path))
    prompt = template.render(prompts_json=prompts_json, sample_size=BATCH_SIZE)

    backend, model_name = load_backend(
        model=args.model, temperature=args.temperature, max_tokens=6400
    )
    wrapper = load_wrapper(model_name, "none", args.system_message)

    generated_prompts = {}
    num_remaining = args.sample_size
    while len(generated_prompts) < args.sample_size:
        wrapper.reset()
        response = wrapper.parse(backend.run(Request(prompt=wrapper.prompt(prompt))))
        batch = dict(list(json.loads(response).items())[:num_remaining])
        generated_prompts |= batch

        num_remaining = args.sample_size - len(generated_prompts)
        logging.info(f"{num_remaining} prompts remaining to be generated")

    logging.info(f"Generated {len(generated_prompts)} prompts")
    save_experiment_json("generated_prompts", generated_prompts)


if __name__ == "__main__":
    main()
