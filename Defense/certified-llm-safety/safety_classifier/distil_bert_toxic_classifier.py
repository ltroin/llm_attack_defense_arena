import re
import warnings
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# specify GPU
# device = 'cpu'  # torch.device("cuda")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_text(filename):
    string = []
    with open(filename, "r") as f:
        full_text = f.read()
        for l in re.split(r"\n", full_text):
            string.append(l)
    return pd.DataFrame(string)


seed = 912

safe_prompt_train = read_text("../data/safe_prompts_train_insertion_erased.txt")
harm_prompt_train = read_text("../data/harmful_prompts_train.txt")
# safe_prompt_train = read_text("../data/safe_prompts_train.txt")
# harm_prompt_train = read_text("../data/harmful_prompts_train.txt")
prompt_data_train = pd.concat([safe_prompt_train, harm_prompt_train], ignore_index=True)
prompt_data_train['Y'] = pd.Series(np.concatenate([np.ones(safe_prompt_train.shape[0]), np.zeros(harm_prompt_train.shape[0])])).astype(int)

safe_prompt_test = read_text("../data/safe_prompts_test_insertion_erased.txt")
harm_prompt_test = read_text("../data/harmful_prompts_test.txt")
# safe_prompt_test = read_text("../data/safe_prompts_test.txt")
# harm_prompt_test = read_text("../data/harmful_prompts_test.txt")
prompt_data_test = pd.concat([safe_prompt_test, harm_prompt_test], ignore_index=True)
prompt_data_test['Y'] = pd.Series(np.concatenate([np.ones(safe_prompt_test.shape[0]), np.zeros(harm_prompt_test.shape[0])])).astype(int)

# split train dataset into train, validation and test sets
train_text, val_text, train_labels, val_labels = train_test_split(prompt_data_train[0], 
								prompt_data_train['Y'], 
								random_state=seed, 
								test_size=0.2,
								stratify=prompt_data_train['Y'])

test_text = prompt_data_test[0]
test_labels = prompt_data_test['Y']

#val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels, 
#                                                                random_state=seed, 
#                                                                test_size=0.5, 
#                                                                stratify=temp_labels)


from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# pass the pre-trained DistilBert to our define architecture
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

# tokenize and encode sequences in the training set
tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length = 25,
    pad_to_max_length=True,
    truncation=True
)

# tokenize and encode sequences in the validation set
tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length = 25,
    pad_to_max_length=True,
    truncation=True
)

# tokenize and encode sequences in the test set
tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length = 25,
    pad_to_max_length=True,
    truncation=True
)

## convert lists to tensors

train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())
# import pdb; pdb.set_trace()
val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels.tolist())

test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels.tolist())

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

#define a batch size
batch_size = 32

# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)

# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)

# dataLoader for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# wrap tensors
val_data = TensorDataset(val_seq, val_mask, val_y)

# sampler for sampling the data during training
val_sampler = SequentialSampler(val_data)

# dataLoader for validation set
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)


# push the model to GPU
model = model.to(device)

# optimizer from hugging face transformers
from transformers import AdamW

# define the optimizer
optimizer = AdamW(model.parameters(), lr = 1e-5)          # learning rate

from sklearn.utils.class_weight import compute_class_weight

#compute the class weights
class_weights = compute_class_weight(class_weight = 'balanced', classes = np.unique(train_labels), y = train_labels.to_numpy())

print("Class Weights:",class_weights)

# converting list of class weights to a tensor
weights= torch.tensor(class_weights,dtype=torch.float)

# push to GPU
weights = weights.to(device)

# define the loss function
cross_entropy  = nn.NLLLoss(weight=weights) 

# number of training epochs
epochs = 5

# function to train the model
def train():
  
  model.train()

  total_loss, total_accuracy = 0, 0
  
  # empty list to save model predictions
  total_preds=[]
  
  # iterate over batches
  for step, batch in enumerate(train_dataloader):
    
    # progress update after every 50 batches.
    if step % 50 == 0 and not step == 0:
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

    # push the batch to gpu
    batch = [r.to(device) for r in batch]
 
    sent_id, mask, labels = batch

    # clear previously calculated gradients 
    model.zero_grad()        

    # get model predictions for the current batch
    preds = model(sent_id, mask)[0]

    # compute the loss between actual and predicted values
    loss = cross_entropy(preds, labels)

    # add on to the total loss
    total_loss = total_loss + loss.item()

    # backward pass to calculate the gradients
    loss.backward()

    # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # update parameters
    optimizer.step()

    # model predictions are stored on GPU. So, push it to CPU
    preds=preds.detach().cpu().numpy()

    # append the model predictions
    total_preds.append(preds)

  # compute the training loss of the epoch
  avg_loss = total_loss / len(train_dataloader)
  
  # predictions are in the form of (no. of batches, size of batch, no. of classes).
  # reshape the predictions in form of (number of samples, no. of classes)
  total_preds  = np.concatenate(total_preds, axis=0)

  #returns the loss and predictions
  return avg_loss, total_preds

# function for evaluating the model
def evaluate():
  
  print("\nEvaluating...")
  
  # deactivate dropout layers
  model.eval()

  total_loss, total_accuracy = 0, 0
  
  # empty list to save the model predictions
  total_preds = []

  # iterate over batches
  for step,batch in enumerate(val_dataloader):
    
    # Progress update every 50 batches.
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
    for epoch in range(epochs):
        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
      
        #train model
        train_loss, _ = train()
      
        #evaluate model
        valid_loss, _ = evaluate()
      
        #save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'new_distillbert_saved_weights.pt')
        
        # append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')


#load weights of best model
path = 'new_distillbert_saved_weights.pt'
model.load_state_dict(torch.load(path))
model.save_pretrained("../models", from_pt=True)

model.eval()

# get predictions for test data
with torch.no_grad():
  preds = model(test_seq.to(device), test_mask.to(device))[0]
  preds = preds.detach().cpu().numpy()

preds = np.argmax(preds, axis = 1)
print(f'Testing Accuracy = {100*torch.sum(torch.tensor(preds) == test_y)/test_y.shape[0]}%')
print(classification_report(test_y, preds))
