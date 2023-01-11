import time
import re
from functools import partial
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import optim
from torch.utils.data import DataLoader
import transformers
from transformers import DistilBertModel, DistilBertTokenizer, logging
from model_utils import EmailDataset, EmailClassifier
from model_utils import transformer_collate_fn, epoch_time, train, evaluate, evaluate_acc
logging.set_verbosity_error()

# Read in data
df = pd.read_csv("/content/traindata.csv")

# Create text column of combined data and remove all non alphanumeric characters
df = df.dropna()
df["email"] = df["from"] + " " + df["subject"] + " " + df["body"]
df['email'] = df["email"].apply(lambda text: re.sub(r'[^A-Za-z0-9 ]+', '', text))

# Drop irrelevant features
irrelevant_features = ["company", "from", "subject", "body"]
df.drop(irrelevant_features, inplace=True,axis=1)

# Replace values in status column to a numerical representation
#   SUBMITTED = 0
#   REJECTED = 1
#   IRRELEVANT = 2
df['status'] = df['status'].replace(['SUBMITTED', 'REJECTED', 'IRRELEVANT'], [0, 1, 2])

# Divide data into train, validation, and test datasets
train_ratio = 0.75
validation_ratio = 0.15
test_ratio = 0.10

train_data, test_data, y_train, y_test = train_test_split(df[["email"]], df[['status']], test_size=1 - train_ratio)
val_data, test_data, y_val, y_test = train_test_split(test_data, y_test, test_size=test_ratio/(test_ratio + validation_ratio))

train_data.insert(1, 'status', y_train, True)
test_data.insert(1, 'status', y_test, True)
val_data.insert(1, 'status', y_val, True)
  
# Set up train, validation, and testing datasets
train_dataset = EmailDataset(train_data)
val_dataset = EmailDataset(val_data)
test_dataset = EmailDataset(test_data)

# Load pretrained Distil BERT model and tokenizer and add to custom classification head
bert_model_name = 'distilbert-base-uncased'
bert_model = DistilBertModel.from_pretrained(bert_model_name)
tokenizer = DistilBertTokenizer.from_pretrained(bert_model_name)

# Define models and devices
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EmailClassifier(bert_model)
model.to(device)

# Define hyperparameters
BATCH_SIZE = 32
LR = 1e-5
N_EPOCHS = 3
CLIP = 1.0

# Create pytorch dataloaders from train_dataset, val_dataset, and test_datset
train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE,collate_fn=partial(transformer_collate_fn, tokenizer=tokenizer), shuffle = True)
val_dataloader = DataLoader(val_dataset,batch_size=BATCH_SIZE,collate_fn=partial(transformer_collate_fn, tokenizer=tokenizer))
test_dataloader = DataLoader(test_dataset,batch_size=BATCH_SIZE,collate_fn=partial(transformer_collate_fn, tokenizer=tokenizer))

# Train and evaluate model
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=N_EPOCHS*len(train_dataloader))
train_loss = evaluate(model, train_dataloader, device)
train_acc = evaluate_acc(model, train_dataloader, device)
valid_loss = evaluate(model, val_dataloader, device)
valid_acc = evaluate_acc(model, val_dataloader, device)
print(f'Initial Train Loss: {train_loss:.3f}')
print(f'Initial Train Acc: {train_acc:.3f}')
print(f'Initial Valid Loss: {valid_loss:.3f}')
print(f'Initial Valid Acc: {valid_acc:.3f}')

for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss = train(model, train_dataloader, optimizer, device, CLIP, scheduler)
    end_time = time.time()
    train_acc = evaluate_acc(model, train_dataloader, device)
    valid_loss = evaluate(model, val_dataloader, device)
    valid_acc = evaluate_acc(model, val_dataloader, device)
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\tTrain Acc: {train_acc:.3f}')
    print(f'\tValid Loss: {valid_loss:.3f}')
    print(f'\tValid Acc: {valid_acc:.3f}')

# Test model and get accuracy
test_loss = evaluate(model, test_dataloader, device)
test_acc = evaluate_acc(model, test_dataloader, device)
print(f'Test Loss: {test_loss:.3f}')
print(f'Test Acc: {test_acc:.3f}')

# Saving model into checkpoint.pth file
checkpoint = {'model': model,
          'state_dict': model.state_dict(),
          'optimizer' : optimizer.state_dict()}

torch.save(checkpoint, 'checkpoint.pth')