import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW, get_scheduler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
from predict import RoBERTaPredictor
def load_data(file_path):
    texts, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        datasets = json.load(file)
        for line in datasets:
            # data = json.loads(line)
            texts.append(line['response'])
            labels.append(line['label'])
    return texts, labels

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(int(label), dtype=torch.long)
        }

def evaluate_model(model, data_loader, device):
    model.eval()  # set model to evaluation mode
    total_loss, total_accuracy = 0, 0
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            model_inputs = {k: v.to(device) for k, v in batch.items() if k != 'text'}
            outputs = model(**model_inputs)
            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits.detach().cpu().numpy()
            label_ids = batch['labels'].to('cpu').numpy()
            predictions.extend(np.argmax(logits, axis=1).flatten())
            true_labels.extend(label_ids.flatten())

    avg_loss = total_loss / len(data_loader)
    avg_accuracy = accuracy_score(true_labels, predictions)
    return avg_loss, avg_accuracy

tokenizer = RobertaTokenizer.from_pretrained('hubert233/GPTFuzz')
model = RobertaForSequenceClassification.from_pretrained('hubert233/GPTFuzz', num_labels=2)

MAX_LEN = 512
BATCH_SIZE = 5
EPOCHS =3
LEARNING_RATE = 2e-5
FILE_PATH = './Results/manual_check_prompts.json'

texts, labels = load_data(FILE_PATH)
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    texts, labels, test_size=0.3, random_state=42)

val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=42)


train_dataset = CustomDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = CustomDataset(val_texts, val_labels, tokenizer, MAX_LEN)
test_dataset = CustomDataset(test_texts, test_labels, tokenizer, MAX_LEN) 

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE) 

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * 0.1)
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        model_inputs = {k: v.to(device) for k, v in batch.items() if k != 'text'}
        outputs = model(**model_inputs)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {avg_train_loss}")

    avg_val_loss, avg_val_accuracy = evaluate_model(model, val_loader, device)
    print(f"Epoch {epoch+1}/{EPOCHS}, Validation Loss: {avg_val_loss}, Validation Accuracy: {avg_val_accuracy}")

avg_test_loss, avg_test_accuracy = evaluate_model(model, test_loader, device)
print(f"Test Loss: {avg_test_loss}, Test Accuracy: {avg_test_accuracy}")
if not os.path.exists("./models/custom"):
    os.makedirs("./models/custom")
model_path = "./models/custom"

model.save_pretrained(model_path)

tokenizer.save_pretrained(model_path)


## for saving the fintuned_version result
model = RoBERTaPredictor('./models/custom', device='cuda:0')
evaluator_res = []
tp_roberta, fp_roberta, fn_roberta,tn_roberta = 0, 0, 0,0
test_labels = test_dataset.labels
test_texts = test_dataset.texts
for label,response in zip(test_labels,test_texts):
    prediction = model.predict(response)[0]
    if prediction == 1 and label== '1':
        tp_roberta += 1
    elif prediction == 1 and label == '0':
        fp_roberta += 1
    elif prediction == 0 and label== '1':
        fn_roberta += 1
    elif prediction == 0 and label == '0':
        tn_roberta += 1
    # break
accuray_roberta = (tp_roberta + tn_roberta) / (tp_roberta + fp_roberta + fn_roberta + tn_roberta)
precision_roberta = tp_roberta / (tp_roberta + fp_roberta) if (tp_roberta + fp_roberta) > 0 else 0
recall_roberta = tp_roberta / (tp_roberta + fn_roberta) if (tp_roberta + fn_roberta) > 0 else 0
f1_roberta = 2 * (precision_roberta * recall_roberta) / (precision_roberta + recall_roberta) if (precision_roberta + recall_roberta) > 0 else 0
evaluator_res.append({'model':'roberta','precision':precision_roberta,'recall':recall_roberta,'f1':f1_roberta,'accuracy':accuray_roberta})
with open('./Results/evaluators/evaluator_res_roberta_finetuned).json','w') as f:
    json.dump(evaluator_res,f,indent=4)



