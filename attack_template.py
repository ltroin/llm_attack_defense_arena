import os
import argparse
import json
import models
import sys
from pathlib import Path

from global_config import get_config
from global_utils import load_data, some_processing, setup_environment, reset_sys_path

original_sys_path = setup_environment()
CONFIG = get_config()
MAX_ALLOWED_ITERATION_PER_QUESTION = CONFIG["MAX_ALLOWED_ITERATION_PER_QUESTION"]
REPEAT_TIME_PER_QUESTION = CONFIG["REPEAT_TIME_PER_QUESTION"]
reset_sys_path(original_sys_path)


def initialize_model(model_path):
    """Initialize the model based on the given path."""
    if 'llama' in model_path:
        return models.LocalVLLM(model_path), 'llama-2', 'llama', False
    elif 'gpt-3.5' in model_path or 'gpt-4' in model_path:
        model_name = 'gpt-3.5' if '3.5' in model_path else 'gpt-4'
        return models.OpenAILLM(model_path), model_name, 'gpt', True
    elif 'vicuna' in model_path:
        return models.LocalVLLM(model_path), 'vicuna', 'vicuna', False
    else:
        raise ValueError("Unsupported model. Supports only vicuna, llama-2, gpt-3.5, and gpt-4.")


def save_results(directory_name, model_name, method_name, results):
    """Save the generated results to a file."""
    results_path = Path(f"../../../Results/{directory_name}")
    results_path.mkdir(parents=True, exist_ok=True)
    with open(results_path / f"{method_name}_{model_name}.json", 'w') as f:
        json.dump(results, f, indent=4)


def main(model_path, method_name, num_steps=10):
    """
    The default value usage of generate method is as follows:
    output = local_model.generate(local_model.create_conv_prompt(prompt), max_tokens=100,temperature=1.0,top_p=1.0,frequency_penalty=0.0,presence_penalty=0.0,n=1)
    """
    local_model, model_name, directory_name, openAI_model = initialize_model(model_path)

    data = load_data()
    finished_questions = set()
    results = []

    for question in data:
        current_iteration, current_repeat = 0, 0
        #delete the while loop if running for template based attacks
        while current_repeat < REPEAT_TIME_PER_QUESTION and current_iteration < MAX_ALLOWED_ITERATION_PER_QUESTION:
            current_repeat += 1
            for _ in range(num_steps):
                if question in finished_questions or current_iteration >= MAX_ALLOWED_ITERATION_PER_QUESTION:
                    finished_questions.add(question)
                    current_repeat = REPEAT_TIME_PER_QUESTION
                    break
                current_iteration += 1
                # Your attack logic goes here.
                # prompt, output = your_attack_logic(...)
                # Dummy placeholders for actual logic
                prompt, output = "TODO: Your attack logic here", "Attack output"

                record = {
                    'prompt': prompt, # mandatory field
                    'response': output, # mandatory field
                    'question': question, # mandatory field
                    "iteration": current_iteration,
                    # Include any other information here
                }
                results.append(record)
        

    save_results(directory_name, model_name, method_name, results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to attack AI models.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model to be attacked.")
    parser.add_argument("--method_name", type=str, required=True, help="Name of the attack method.")
    parser.add_argument("--num_steps", type=int, required=True, help="Number of steps to run the attack.")

    args = parser.parse_args()
    main(args.model_path, args.method_name)
