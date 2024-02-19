# Script to count the number of tokens in each line of a file.

from transformers import DistilBertTokenizer, AutoTokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_file', type=str, help='Input file to count tokens in.')

args = parser.parse_args()

min = -1
max = -1
avg = -1
total = 0

# typ = 'Llama'
typ = 'DistilBERT'

if typ == 'DistilBERT':
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
elif typ == 'Llama':
    model = "meta-llama/Llama-2-7b-chat-hf"
    # commit_id = "main"        # to use the latest version
    commit_id = "08751db2aca9bf2f7f80d2e516117a53d7450235"      # to reproduce the results in our paper

    print(f'Loading model {model}...')
    tokenizer = AutoTokenizer.from_pretrained(model, revision=commit_id)
    
with open(args.input_file, 'r') as f:
    for line in f:
        tokens = len(tokenizer.encode(line.strip()))
        if total == 0:
            min = tokens
            max = tokens
            avg = tokens
        else:
            if tokens < min:
                min = tokens
            if tokens > max:
                max = tokens
            avg = (avg * total + tokens) / (total + 1)
        
        total += 1

print('Min: {}'.format(min))
print('Max: {}'.format(max))
print('Avg: {}'.format(avg))
print('Total lines: {}'.format(total))