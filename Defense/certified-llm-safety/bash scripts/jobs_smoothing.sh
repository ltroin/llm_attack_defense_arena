#!/bin/sh

# Evaluation script for smoothing certificates
for max_erase in 10 20 30 # 5 10 15 20
do
    python main.py --num_prompts 50 --max_erase $max_erase --eval_type smoothing
done