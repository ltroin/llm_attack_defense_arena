#!/bin/sh

for i in {1..10}
do
    echo "Run $i:"
    python main.py --eval_type harmful --num_prompts 1000
done