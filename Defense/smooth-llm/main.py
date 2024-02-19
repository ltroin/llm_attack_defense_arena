import os
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import argparse

import lib.perturbations as perturbations
import lib.defenses as defenses
import lib.attacks as attacks
import lib.language_models as language_models
import lib.model_configs as model_configs
from collections import defaultdict
import json
import time
# torch.cuda.memory._record_memory_history(enabled=True,trace_alloc_max_entries=100000)

def main(args):

    # Create output directories
    # os.makedirs(args.results_dir, exist_ok=True)
    model_name = ""
    if 'vicuna' in args.target_model:
        model_name = 'vicuna'
        directory_name='vicuna'
    elif 'llama' in args.target_model:
        model_name = 'llama-2'
        directory_name='llama'
    elif 'gpt-3.5' in args.target_model:
        model_name = 'gpt-3.5-turbo'
        directory_name='gpt'
    elif 'gpt-4' in args.target_model:
        model_name = 'gpt4'
    else:
        model_name = 'unknown'
    # Instantiate the targeted LLM
    if "gpt" not in args.target_model:
        config = model_configs.MODELS[args.target_model]
        target_model = language_models.LocalVLLM(
            model_path=config['model_path'],
            # tokenizer_path=config['tokenizer_path'],
            # conv_template_name=config['conversation_template'],
            # device='cuda:0'
        )
    else:
        target_model = language_models.OpenAILLM(
            model_path=args.target_model,
        )

    # Create SmoothLLM instance
    defense = defenses.SmoothLLM(
        target_model=target_model,
        pert_type=args.smoothllm_pert_type,
        pert_pct=args.smoothllm_pert_pct,
        num_copies=args.smoothllm_num_copies,
        benign_check=args.benign_check
    )

    # Create attack instance, used to create prompts
    if 'gpt' in model_name:
        name = 'gpt-3.5-turbo'
    elif 'vicuna' in model_name:
        name = 'vicuna'
    elif 'llama' in model_name:
        name = 'llama-2'
    if args.benign_check:
        args.attack_logfile = f"../../Results/alpaca/alpaca_benign_{name}.json"


    attack = vars(attacks)[args.attack](
        logfile=args.attack_logfile,
        target_model=target_model,
        benign_check=args.benign_check,
    )
    success_jailbreak = defaultdict(list)
    failed_jailbreak = defaultdict(list)
    comprehensive_results = defaultdict(list)
    last_file_name = ""
    if not args.attack == 'Universal':
        jailbroken_results = []
        for i, prompt in tqdm(enumerate(attack.prompts)):
            input,output = defense(prompt)
            jb = defense.is_jailbroken(output)
            jailbroken_results.append(jb)
    else:
        for file_name,prompts in tqdm(attack.file_name_prompts.items()):
            # if file_name != "Parameters":
            #     continue
            print(f"we are processing {file_name}")
            mutated_inputs= []
            jailbroken_results = []
            outputs = []
            for i, prompt in enumerate(prompts):
                print(f"we are processing {i}th prompt of {file_name}")
                #could add this permutation version or not add
                input,output = defense(prompt,file_name)
                mutated_inputs.append(input)
                outputs.append(output)
                jb = defense.is_jailbroken(output)
                jailbroken_results.append(jb)
                # break
            # torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
            # torch.cuda.memory._record_memory_history(enabled=None)

            
            for prompt, jb, output in zip(prompts, jailbroken_results, outputs):
                #NOTE if jailbroken, meaning, it does not generate enough number of responses that fall in prefix set
                #the benign question would have this behavior. therefore if it is jailbroken, we should think it pass the filter.
                if args.benign_check:
                    prompt.original_prompt[1]=output
                else:
                    prompt.original_prompt[2]=output
                if jb:
                    if args.benign_check:
                        success_jailbreak[file_name].append([prompt.original_prompt[0],output])
                    else:
                        success_jailbreak[file_name].append(prompt.original_prompt)

                    # if args.benign_check:
                    #     success_questions[file_name].append(prompt.perturbable_prompt[0])
                    # else:
                    #     success_questions[file_name].append(prompt.perturbable_prompt[1])
                else:
                    failed_jailbreak[file_name].append(prompt.original_prompt)

                comprehensive_results[file_name].append(prompt.original_prompt)

            

        for file_name,prompts in success_jailbreak.items():
            last_file_name = file_name
            if args.benign_check:
                total_question_set = set(pair.original_prompt[0] for pair in attack.file_name_prompts[file_name])
                success_questions_set = set(pair[0] for pair in success_jailbreak[file_name])
                success_jailbreak[file_name].append(f"The benign question pass rate of is {len(success_questions_set)}/{len(total_question_set)}")

            else:
                total_question_set = set(pair.original_prompt[1] for pair in attack.file_name_prompts[file_name])
                # total_prompts_set = set(pair.original_prompt[0] for pair in attack.file_name_prompts[file_name])
                success_questions_set = set(pair[1] for pair in success_jailbreak[file_name])
                # success_jailbreak_set = set(pair[0] for pair in success_jailbreak[file_name])
                success_jailbreak[file_name].append(f"The Malicious prompt pass rate is {len(success_jailbreak[file_name])}/{len(attack.file_name_prompts[file_name])}")
                success_jailbreak[file_name].append(f"The Malicious question pass rate is {len(success_questions_set)}/{len(total_question_set)}")

    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    success_jailbreak[last_file_name].append(f"Elapsed time: {elapsed_minutes} minutes")

    if not os.path.exists(f"../../Results/defense/{directory_name}"):
            os.makedirs(f"../../Results/defense/{directory_name}")

    if args.benign_check:
        with open(f"../../Results/defense/{directory_name}/smoothllm_{model_name}_benign.json",'w') as f:
            json.dump(success_jailbreak, f, indent=4)
        with open (f"../../Results/defense/{directory_name}/smoothllm_comprehensive_{model_name}_benign.json",'w') as f:
            json.dump(comprehensive_results,f,indent=4)
    else:
        with open(f"../../Results/defense/{directory_name}/smoothllm_{model_name}.json",'w') as f:
            json.dump(success_jailbreak, f, indent=4)
        with open (f"../../Results/defense/{directory_name}/smoothllm_comprehensive_{model_name}.json",'w') as f:
                json.dump(comprehensive_results,f,indent=4)
    

                    

    # print(f'We made {num_errors} errors')

    # # Save results to a pandas DataFrame
    # summary_df = pd.DataFrame.from_dict({
    #     'Number of smoothing copies': [args.smoothllm_num_copies],
    #     'Perturbation type': [args.smoothllm_pert_type],
    #     'Perturbation percentage': [args.smoothllm_pert_pct],
    #     'JB percentage': [np.mean(jailbroken_results) * 100],
    #     'Trial index': [args.trial]
    # })
    # summary_df.to_pickle(os.path.join(
    #     args.results_dir, 'summary.pd'
    # ))
    # print(summary_df)


if __name__ == '__main__':
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--results_dir',
    #     type=str,
    #     default='./results'
    # )
    # parser.add_argument(
    #     '--trial',
    #     type=int,
    #     default=0
    # )

    # Targeted LLM
    parser.add_argument(
        '--target_model',
        type=str,
        default='gpt-3.5-turbo',
        choices=['vicuna', 'llama','gpt-3.5-turbo','gpt-4-turbo']
    )

    # Attacking LLM
    parser.add_argument(
        '--attack',
        type=str,
        default='Universal',
        choices=['GCG', 'PAIR', 'Universal']
    )
    parser.add_argument(
        '--attack_logfile',
        type=str,
        default='../../Results/vicuna/Merged_vicuna.json'
    )

    # SmoothLLM
    parser.add_argument(
        '--smoothllm_num_copies',
        type=int,
        default=10,
    )
    parser.add_argument(
        '--smoothllm_pert_pct',
        type=int,
        default=10
    )
    parser.add_argument(
        '--smoothllm_pert_type',
        type=str,
        default='RandomSwapPerturbation',
        choices=[
            'RandomSwapPerturbation',
            'RandomPatchPerturbation',
            'RandomInsertPerturbation'
        ]
    )
    parser.add_argument(
        '--benign_check',
        type=str,
        default="false"
    )
    

    args = parser.parse_args()
    args.benign_check = args.benign_check.lower() == "true"
    
    start_time = time.time()
    args.start_time = start_time
    main(args)