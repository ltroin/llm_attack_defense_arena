## Erase and Check via approximate L0 optimization
## argmax_{mask} loss(prompt \odot mask) s.t. mak has least possible sparsity 

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import argparse
import time

from defenses import progress_bar

def is_harmful(prompt, model, tokenizer, 
               num_iters=50, init_temp=0.5, reg_const=0.1):
    
    """
    Wrap around the adversarial masking procedure as a "is_harmful" function
    """
    
    tokens = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    model_output = model(tokens)
    output_class = model_output[0].argmax().item()
    if output_class == 0:
        # if prompt is initially labelled harmful
        # return harmful
        return True
    else:
        # if prompt is initially labelled safe
        # apply adversarial masking to find it can be 
        # converted to be labelled harmful

        masked_prompt = grad_ec(prompt, model, tokenizer,
                        num_iters=num_iters, init_temp=init_temp, reg_const=reg_const)
    
        tokens = torch.tensor(tokenizer.encode(masked_prompt)).unsqueeze(0).to(device)
        model_output = model(tokens)
        #print("Masked Prediction: " + ("safe" if model_output[0].argmax().item() == 1 else "harmful"))
        output_class = model_output[0].argmax().item()
        return (output_class == 0)

def grad_ec(prompt, model, tokenizer, word_embeddings, num_iters=50,
            init_temp=0.5, reg_const=1e-3):
    """
    This function implements GradEC -- a gradient based approximation of the erase-and-check procedure.
    It optimizes a mask over the input prompt to erase tokens and check if the resulting
    prompt is harmful or not. If the resulting sequence is harmful, it returns True, else False. It also
    returns the erased prompt.

    Args:
        prompt: Input prompt
        model: Safety classifier
        tokenizer: Tokenizer for the safety classifier
        word_embeddings: Embedding layer of the safety classifier
        num_iters: Number of iterations
        init_temp: Initial temperature value for the sigmoid function
        reg_const: Regularization constant enforcing masks to have small sparsity
    Returns:
        harmful: Whether the input prompt is harmful or not
        masked_prompt: input prompt with some tokens masked out 
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize adversarial sequence and convert to tokens
    tokens = torch.tensor(tokenizer.encode(prompt,max_length=500, truncation=True)).unsqueeze(0).to(device)

    # Initialize to all zeros
    mask_logits = torch.zeros_like(tokens).float()
    mask_logits = torch.nn.Parameter(mask_logits.requires_grad_().to(device))
    optimizer = torch.optim.SGD(params=[mask_logits], lr=1e-1, momentum=0.9, weight_decay=0.)

    # word_embeddings = model.distilbert.embeddings.word_embeddings

    temp = max(init_temp, 1e-8) # Avoid division by zero

    for i in range(num_iters):

        optimizer.zero_grad()

        # Critical heuristic: Decay temperature over time
        # as temp -> 0; mask becomes binary
        temp = init_temp / (i+1)

        mask_sigmoid = torch.sigmoid(mask_logits / temp)

        # Erased prompt
        binary_mask = (mask_sigmoid >= 0.5).long()
        # binary_mask = (mask_sigmoid >= 0.5).float()
        masked_tokens = binary_mask * tokens
        # masked_tokens = torch.round(binary_mask * tokens).long()
        masked_prompt = tokenizer.decode((masked_tokens)[0][1:-1])

        # If erased prompt is harmful, return True
        model_output = model(torch.tensor(tokenizer.encode(masked_prompt,max_length=500, truncation=True)).unsqueeze(0).to(device)) # Evaluates model on erased prompt
        if model_output.logits[0].argmax().item() == 0:
            return True, masked_prompt

        # Regularize mask such that most of the probabilities are close to one
        regularizer = (torch.sigmoid(mask_logits) - 1).pow(2).mean()

        embeddings = word_embeddings(tokens)
        embeddings = mask_sigmoid.unsqueeze(2) * embeddings
        
        # Class 0 is harmful
        output = model(inputs_embeds=embeddings, labels=torch.tensor([0]).to(device)) 

        loss = output.loss + reg_const * regularizer
        loss.backward() 
        optimizer.step()

    mask_sigmoid = torch.sigmoid(mask_logits / temp)
    # binary_mask = (mask_sigmoid >= 0.5).float()
    # masked_tokens = torch.round(binary_mask * tokens).long()
    binary_mask = (mask_sigmoid >= 0.5).long()
    masked_tokens = binary_mask * tokens
    masked_prompt = tokenizer.decode((masked_tokens)[0][1:-1])
   
    # If erased prompt is harmful, return True
    model_output = model(torch.tensor(tokenizer.encode(masked_prompt)).unsqueeze(0).to(device)) # Evaluates model on erased prompt
    if model_output.logits[0].argmax().item() == 0:
        return True, masked_prompt
    else:
        return False, masked_prompt


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Adversarial masks for the safety classifier.')
    parser.add_argument('--prompts_file', type=str, default='data/adversarial_prompts_t_21.txt', help='File containing prompts')
    parser.add_argument('--num_iters', type=int, default=50, help='Number of iterations')

    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model and tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

    # Load model weights
    model_wt_path = 'models/distillbert_saved_weights.pt'
    
    model.load_state_dict(torch.load(model_wt_path))
    model.to(device)
    model.eval()

    prompts_file = args.prompts_file
    num_iters = args.num_iters

    # Load prompts
    prompts = []
    with open(prompts_file, 'r') as f:
        for line in f:
            prompts.append(line.strip())

    print("Loaded " + str(len(prompts)) + " prompts.")
    list_of_bools = []
    start_time = time.time()

    for num_done, input_prompt in enumerate(prompts):
        
        decision, masked_prompt = grad_ec(input_prompt, model, tokenizer, model.distilbert.embeddings.word_embeddings,
                    num_iters=num_iters, init_temp=float(num_iters/100), reg_const=1e-3)
        list_of_bools.append(decision)

        # print("ORIGINAL PROMPT: " + input_prompt)
        # print("Masked Prompt: " + masked_prompt)
        # print("Prediction: " + ("harmful" if decision else "safe"))
        # input('Press enter to continue')
        
        percent_harmful = (sum(list_of_bools) / len(list_of_bools)) * 100.
        current_time = time.time()
        elapsed_time = current_time - start_time
        time_per_prompt = elapsed_time / (num_done + 1)

        print("Checking safety... " + progress_bar((num_done + 1) / len(prompts)) \
            + f' Detected harmful = {percent_harmful:5.1f}%' \
            + f' Time/prompt = {time_per_prompt:5.1f}s', end="\r")
        
    print("")