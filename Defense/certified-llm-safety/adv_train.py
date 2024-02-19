import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import argparse
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import re
from transformers import AdamW
from sklearn.utils.class_weight import compute_class_weight
from gcg import gcg_suffix
from prompt_dataset import PromptDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 912

def read_text(filename):
    string = []
    with open(filename, "r") as f:
        full_text = f.read()
        for l in re.split(r"\n", full_text):
            string.append(l)
    return pd.DataFrame(string)

tokenizer_adv = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', cache_dir="/n/holyscratch01/hlakkaraju_lab/Lab/aaronli/models")
model_adv = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', cache_dir="/n/holyscratch01/hlakkaraju_lab/Lab/aaronli/models")

model_wt_path = "../models/new_distilbert_saved_weights.pt"
model_adv.load_state_dict(torch.load(model_wt_path))
model_adv.to(device)
model_adv.eval()

num_adv = 10
generate_test_harmful_adv = True
generate_test_safe_adv = True

adv_source_file = "./data/harmful_prompts_test.txt"

# Load prompts
prompts = []
with open(adv_source_file, 'r') as f:
    for line in f:
        prompts.append(line.strip())

print("Loaded " + str(len(prompts)) + " prompts.")

# Open file to write prompts
harmful_adv_saved_file = "./data/harmful_prompts_test_suffix_adv.txt"

if generate_test_harmful_adv:
    f = open(harmful_adv_saved_file, 'w')
    num_success = 0
    for input in prompts:
        # print("ORG PROMPT: " + input)

        adv_prompt = gcg_suffix(input, model_adv, tokenizer_adv, model_adv.distilbert.embeddings.word_embeddings,
                                model_adv.distilbert.embeddings.word_embeddings.weight,
                                num_adv=num_adv)

        
        # print("ADV PROMPT: " + adv_prompt)
        tokens = torch.tensor(tokenizer_adv.encode(adv_prompt)).unsqueeze(0).to(device)
        model_output = model_adv(tokens)
        pred = "safe" if model_output[0].argmax().item() == 1 else "harmful"
        #print("Prediction: " + pred)
        if pred == "safe":
            num_success += 1
        f.write(adv_prompt + '\n')
        f.flush()

    f.close()
    print(f"Attack success rate: {num_success / len(prompts)}")

if generate_test_safe_adv:
    adv_source_file_safe = "./data/safe_prompts_test.txt"
    safe_test_prompts = []
    with open(adv_source_file_safe, 'r') as f:
        for line in f:
            safe_test_prompts.append(line.strip())
    safe_adv_saved_file = "./data/safe_prompts_test_suffix_adv.txt"
    f = open(safe_adv_saved_file, 'w')
    for prompt in safe_test_prompts:
        #print(prompt)
        encoded_prompt = tokenizer_adv.encode_plus(prompt)['input_ids']
        erased_encoded = encoded_prompt[1:-5]
        erased_prompt = tokenizer_adv.decode(erased_encoded)
        #print(erased_prompt)
        f.write(erased_prompt + '\n')
        f.flush()
    f.close()

safe_prompts_train = read_text("./data/safe_prompts_train.txt")
harmful_prompts_train = read_text("./data/harmful_prompts_train.txt")
safe_prompts_test = read_text("./data/safe_prompts_test.txt")
harmful_prompts_test = read_text("./data/harmful_prompts_test.txt")
safe_prompts_test_adv = read_text(safe_adv_saved_file)
harmful_prompts_test_adv = read_text(harmful_adv_saved_file)
prompt_data_train = pd.concat([safe_prompts_train, harmful_prompts_train], ignore_index=True)
prompt_data_train['Y'] = pd.Series(np.concatenate([np.ones(safe_prompts_train.shape[0]), np.zeros(harmful_prompts_train.shape[0])])).astype(int)
prompt_data_train.to_csv("./data/train_adv.csv")
prompt_data_test = pd.concat([safe_prompts_test, safe_prompts_test_adv, harmful_prompts_test, harmful_prompts_test_adv], ignore_index=True)
prompt_data_test['Y'] = pd.Series(np.concatenate([np.ones(safe_prompts_test.shape[0] + safe_prompts_test_adv.shape[0]), np.zeros(harmful_prompts_test.shape[0] + harmful_prompts_test_adv.shape[0])])).astype(int)
prompt_data_test.to_csv("./data/test_adv.csv")
train_text, val_text, train_labels, val_labels = train_test_split(prompt_data_train[0], 
								prompt_data_train['Y'], 
								random_state=seed, 
								test_size=0.2,
								stratify=prompt_data_train['Y'])

test_text = prompt_data_test[0]
test_labels = prompt_data_test['Y']

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length = 25,
    pad_to_max_length=True,
    truncation=True
)

tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length = 25,
    pad_to_max_length=True,
    truncation=True
)

tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length = 25,
    pad_to_max_length=True,
    truncation=True
)

train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
# train_mask: (train_size, 25)
train_y = torch.tensor(train_labels.tolist())

val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels.tolist())

test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels.tolist())

batch_size = 16
train_dataset = PromptDataset(train_text.to_list(), train_seq, train_mask, train_y)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
print(f'Training data size: {len(train_dataset)}')

val_dataset = TensorDataset(val_seq, val_mask, val_y)
val_sampler = SequentialSampler(val_dataset)
val_dataloader = DataLoader(val_dataset, sampler = val_sampler, batch_size=batch_size)
print(f'Validation data size: {len(val_dataset)}')

test_dataset = TensorDataset(test_seq, test_mask, test_y)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler = test_sampler, batch_size=batch_size)
print(f'Test data size: {len(test_dataset)}')

model = model.to(device)
optimizer = AdamW(model.parameters(), lr = 1e-5)
class_weights = compute_class_weight(class_weight = 'balanced', classes = np.unique(train_labels), y = train_labels.to_numpy())
weights= torch.tensor(class_weights,dtype=torch.float)
weights = weights.to(device)
cross_entropy  = nn.NLLLoss(weight=weights) 

num_epochs = 5
erase_length = 5

def train():
    model.train()
    total_loss, total_accuracy = 0, 0
    total_preds=[]
    num_total = 0
    num_correct = 0
    for step, batch in enumerate(train_dataloader):
        # push the batch to gpu
        # (batch_size, 25), (batch_size, 25), (batch_size)
        original_prompts, sent_id, mask, labels = batch
        sent_id = sent_id.to(device)
        mask = mask.to(device)
        labels = labels.to(device)
        #print(sent_id)
        #print(labels)
        for i in range(batch_size):
            # harmful
            if labels[i] == 0:
                #continue
                adv_prompt = gcg_suffix(original_prompts[i], model_adv, tokenizer, model.distilbert.embeddings.word_embeddings,
                    model.distilbert.embeddings.word_embeddings.weight,
                    num_adv=num_adv)
                token_harmful = tokenizer.encode_plus(adv_prompt, max_length=25,
                    pad_to_max_length=True,
                    truncation=True)
                new_label = labels[i].clone()
                new_seq = torch.tensor(token_harmful['input_ids']).to(device)
                new_mask = torch.tensor(token_harmful['attention_mask']).to(device)
                labels = torch.cat([labels, new_label.unsqueeze(dim=0)])
                sent_id = torch.cat([sent_id, new_seq.unsqueeze(dim=0)])
                mask = torch.cat([mask, new_mask.unsqueeze(dim=0)])

            # safe
            else:
                #continue
                
                new_mask = mask[i].clone()
                length_padding = len(sent_id[i][sent_id[i] == 0])
                new_seq = sent_id[i].clone()
                new_label = labels[i].clone()
                #print(new_seq)
                new_seq[-(length_padding+erase_length):] = 0
                #print(new_seq)
                labels = torch.cat([labels, new_label.unsqueeze(dim=0)])
                #print(labels)
                sent_id = torch.cat([sent_id, new_seq.unsqueeze(dim=0)])
                mask = torch.cat([mask, new_mask.unsqueeze(dim=0)])
                #print(i)
                #print(labels)
                #print(sent_id)

        
        indices = torch.randperm(sent_id.shape[0]).to(device)
        sent_id = torch.index_select(sent_id, 0, indices)
        mask = torch.index_select(mask, 0, indices)
        labels = torch.index_select(labels, 0, indices)
        #print(indices)
        #print(sent_id)
        #print(labels)
        #break
        model.zero_grad()        
        preds = model(sent_id, mask)[0]
        num_total += sent_id.shape[0]
        num_correct += len(torch.argmax(preds, axis=1) == labels)
        loss = cross_entropy(preds, labels)
        total_loss = total_loss + loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        preds=preds.detach().cpu().numpy()
        total_preds.append(preds)

    avg_loss = total_loss / len(train_dataloader)
    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)

    return avg_loss, num_correct / num_total


def evaluate():
  
  print("\nEvaluating...")
  model.eval()

  total_loss, total_accuracy = 0, 0
  total_preds = []

  for step,batch in enumerate(val_dataloader):
    if step % 50 == 0 and not step == 0:
      
      # Calculate elapsed time in minutes.
      # elapsed = format_time(time.time() - t0)
            
      # Report progress.
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))
    # push the batch to gpu
    batch = [t.to(device) for t in batch]

    sent_id, mask, labels = batch

    # deactivate autograd
    with torch.no_grad():
      
      # model predictions
      preds = model(sent_id, mask)[0]

      # compute the validation loss between actual and predicted values
      loss = cross_entropy(preds, labels)

      total_loss = total_loss + loss.item()

      preds = preds.detach().cpu().numpy()

      total_preds.append(preds)

  # compute the validation loss of the epoch
  avg_loss = total_loss / len(val_dataloader) 

  # reshape the predictions in form of (number of samples, no. of classes)
  total_preds  = np.concatenate(total_preds, axis=0)

  return avg_loss, total_preds

# set initial loss to infinite
best_valid_loss = float('inf')

# empty lists to store training and validation loss of each epoch
train_losses=[]
valid_losses=[]
train_flag = True

if train_flag == True:
    # for each epoch
    for epoch in range(num_epochs):
        print('\n Epoch {:} / {:}'.format(epoch + 1, num_epochs))
      
        #train model
        train_loss, train_acc = train()
        print(f"Epoch {epoch} training accuracy: {train_acc}")
        #evaluate model
        #valid_loss, _ = evaluate()
      
        #save the best model
        #if valid_loss < best_valid_loss:
        #    best_valid_loss = valid_loss
        #    torch.save(model.state_dict(), 'new_distillbert_saved_weights.pt')
        
        # append training and validation loss
        train_losses.append(train_loss)
        #valid_losses.append(valid_loss)
        
        print(f'\nTraining Loss: {train_loss:.3f}')
        #print(f'Validation Loss: {valid_loss:.3f}')
        model.eval()

        # get predictions for test data
        with torch.no_grad():
            preds = model(test_seq.to(device), test_mask.to(device))[0]
            preds = preds.detach().cpu().numpy()

        preds = np.argmax(preds, axis = 1)
        print(f'Testing Accuracy = {100*torch.sum(torch.tensor(preds) == test_y)/test_y.shape[0]}%')
        print(classification_report(test_y, preds))

#load weights of best model
#path = 'new_distillbert_saved_weights.pt'
#model.load_state_dict(torch.load(path))
#model.eval()

# get predictions for test data
#with torch.no_grad():
#    preds = model(test_seq.to(device), test_mask.to(device))[0]
#    preds = preds.detach().cpu().numpy()

#preds = np.argmax(preds, axis = 1)
#print(f'Testing Accuracy = {100*torch.sum(torch.tensor(preds) == test_y)/test_y.shape[0]}%')
#print(classification_report(test_y, preds))