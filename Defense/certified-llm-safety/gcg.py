import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from termcolor import colored
import argparse


def gcg_suffix(input, model, tokenizer, word_embeddings, embedding_matrix,
               num_adv=10, num_iters=50, top_k=10, batch_size=100):
    """
    Generates adversarial suffix for the safety classifier using GCG.
    Args:
        input: Input prompt
        model: Safety classifier
        tokenizer: Tokenizer for the safety classifier
        word_embeddings: Embedding layer of the safety classifier
        embedding_matrix: Embedding matrix of the safety classifier
        num_adv: Number of adversarial tokens to append
        num_iters: Number of iterations of GCG
        top_k: Number of top adversarial tokens to consider
        batch_size: Batch size for GCG
    Returns:
        adv_prompt: Input prompt with adversarial suffix
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize adversarial sequence and convert to tokens
    tokens = torch.tensor(tokenizer.encode(input + ('!' * num_adv))).unsqueeze(0).to(device)
    prompt_len = len(tokenizer.decode(torch.tensor(tokenizer.encode(input)).unsqueeze(0).to(device)[0][1:-1]))

    for _ in range(num_iters):
        # Get token embeddings
        embeddings = word_embeddings(tokens)

        # Compute gradient of loss w.r.t. embeddings
        embeddings.retain_grad()
        output = model(inputs_embeds=embeddings, labels=torch.tensor([1]).to(device)) # Class 1 is safe
        loss = output.loss
        (-loss).backward()  # Minimize loss
        gradients = embeddings.grad

        # Dot product of gradients and embedding matrix
        dot_prod = torch.matmul(gradients[0], embedding_matrix.T)

        # Set dot product of [CLS] and [SEP] tokens to -inf
        cls_token_idx = tokenizer.encode('[CLS]')[1]
        sep_token_idx = tokenizer.encode('[SEP]')[1]
        dot_prod[:, cls_token_idx] = -float('inf')
        dot_prod[:, sep_token_idx] = -float('inf')

        # Get top k adversarial tokens
        top_k_adv = (torch.topk(dot_prod, top_k).indices)[-num_adv-1:-1]    # Last token is [SEP]

        # Create a batch of adversarial prompts by uniformly sampling from top k adversarial tokens
        tokens_batch = [tokens.clone().detach()]
        for _ in range(batch_size):
            random_indices = torch.randint(0, top_k, (num_adv,)).to(device)
            selected_tokens = torch.gather(top_k_adv, 1, random_indices.unsqueeze(1))
            batch_item = tokens.clone().detach()
            batch_item[0, -num_adv-1:-1] = selected_tokens.squeeze(1)
            tokens_batch.append(batch_item)

        tokens_batch = torch.cat(tokens_batch, dim=0)

        # Pick batch elements with highest logit
        output = model(inputs_embeds=word_embeddings(tokens_batch))
        max_idx = torch.argmax(output.logits[:, 1])
        tokens = tokens_batch[max_idx].unsqueeze(0)

        adv_prompt = tokenizer.decode(tokens[0][1:-1])
        print("ADV PROMPT: " + adv_prompt[0:prompt_len] + colored(adv_prompt[prompt_len:], 'red'), end='\r')
    
    print()

    return adv_prompt


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate adversarial prompts using GCG.')
    parser.add_argument('--num_adv', type=int, default=10, help='Number of adversarial tokens to append')
    parser.add_argument('--prompts_file', type=str, default='data/harmful_prompts_test.txt', help='File containing prompts')
    parser.add_argument('--model_wt_path', type=str, default='models/distilbert_saved_weights.pt', help='Path to model weights')

    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model and tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

    # Load model weights
    model_wt_path = args.model_wt_path
    model.load_state_dict(torch.load(model_wt_path))
    model.to(device)
    model.eval()

    num_adv = args.num_adv
    prompts_file = args.prompts_file

    # Load prompts
    prompts = []
    with open(prompts_file, 'r') as f:
        for line in f:
            prompts.append(line.strip())

    print("Loaded " + str(len(prompts)) + " prompts.")

    # Open file to write prompts
    f = open('data/adversarial_prompts_t_' + str(num_adv) + '.txt', 'w')
    for input in prompts:
        # print("ORG PROMPT: " + input)

        adv_prompt = gcg_suffix(input, model, tokenizer, model.distilbert.embeddings.word_embeddings,
                                model.distilbert.embeddings.word_embeddings.weight,
                                num_adv=num_adv)

        
        # print("ADV PROMPT: " + adv_prompt)
        tokens = torch.tensor(tokenizer.encode(adv_prompt)).unsqueeze(0).to(device)
        model_output = model(tokens)
        print("Prediction: " + ("safe" if model_output[0].argmax().item() == 1 else "harmful"))
        print()
        f.write(adv_prompt + '\n')
        f.flush()

    f.close()
