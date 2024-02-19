#!/bin/bash

# Evaluation script for suffix attacks with max erase running from 0 to 40 with step 10
for max_erase in 0 10 20 30
do
    python main.py \
        --num_prompts 520 \
        --mode suffix \
        --eval_type safe \
        --max_erase $max_erase \
        --safe_prompts data/safe_prompts.txt \
        # --use_classifier \
        # --model_wt_path models/distillbert_saved_weights.pt \
        # --safe_prompts data/safe_prompts_test.txt \
done