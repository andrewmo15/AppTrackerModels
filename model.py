import time
from functools import partial
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import optim
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer, get_linear_schedule_with_warmup
# from tokenizers.processors import BertProcessing

# Read in data
df = pd.read_csv("data generation/data.csv")

# Create text column of combined data
df["email"] = df["from"] + " " + df["subject"] + " " + df["body"]

# Drop irrelevant features
irrelevant_features = ["company", "from", "subject", "body"]
df.drop(irrelevant_features, inplace=True,axis=1)

# Divide data into train, validation, and test datasets
train_ratio = 0.75
validation_ratio = 0.15
test_ratio = 0.10

train_data, test_data, y_train, y_test = train_test_split(df[["email"]], df[['status']], test_size=1 - train_ratio)
val_data, test_data, y_val, y_test = train_test_split(test_data, y_test, test_size=test_ratio/(test_ratio + validation_ratio))

train_data.insert(1, 'status', y_train, True)
test_data.insert(1, 'status', y_test, True)
val_data.insert(1, 'status', y_val, True)

# Defining torch dataset class for disaster tweet dataset
class EmailDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]

# Set up train, validation, and testing datasets
train_dataset = EmailDataset(train_data)
val_dataset = EmailDataset(val_data)
test_dataset = EmailDataset(test_data)

# Create collate function for email dataset that will tokenize the input emails for use with our BERT models.
def transformer_collate_fn(batch, tokenizer):
    bert_vocab = tokenizer.get_vocab()
    bert_pad_token = bert_vocab['[PAD]']
    sentences, labels, masks = [], [], []
    for data in batch:
        tokenizer_output = tokenizer([data['email']])
        tokenized_sent = tokenizer_output['input_ids'][0]
        mask = tokenizer_output['attention_mask'][0]
        sentences.append(torch.tensor(tokenized_sent))
        labels.append(torch.tensor(data['status']))
        masks.append(torch.tensor(mask))
    sentences = pad_sequence(sentences, batch_first=True, padding_value=bert_pad_token)
    labels = torch.stack(labels, dim=0)
    masks = pad_sequence(masks, batch_first=True, padding_value=0.0)
    return sentences, labels, masks

# Model helper functions:
# Initialize weights for model
def init_weights(m: nn.Module, hidden_size=768):
    k = 1 / hidden_size
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.uniform_(param.data, a=-1*k**0.5, b=k**0.5)
        else:
            nn.init.uniform_(param.data, 0)

# Convert epoch time to readable form
def epoch_time(start_time: int, end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# Count the number of trainable parameters in the model
def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Train a given model, using a pytorch dataloader, optimizer, and scheduler
def train(model, dataloader, optimizer, device, clip: float, scheduler = None):
    model.train()
    epoch_loss = 0
    for batch in dataloader:
        sentences, labels, masks = batch[0], batch[1], batch[2]
        optimizer.zero_grad()
        output = model(sentences.to(device), masks.to(device))
        loss = F.cross_entropy(output, labels.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

# Calculate epoch loss from the model on dataloader
def evaluate(model, dataloader, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            sentences, labels, masks = batch[0], batch[1], batch[2]
            output = model(sentences.to(device), masks.to(device))
            loss = F.cross_entropy(output, labels.to(device))
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

# Calculate prediction accuracy on dataloader
def evaluate_acc(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total = 0
        for _, batch in enumerate(dataloader):
            sentences, labels, masks = batch[0], batch[1], batch[2]
            output = model(sentences.to(device), masks.to(device))
            output = F.softmax(output, dim=1)
            output_class = torch.argmax(output, dim=1)
            total_correct += torch.sum(torch.where(output_class == labels.to(device), 1, 0))
            total += sentences.size()[0]
    return total_correct / total

# Model creation
class EmailClassifier(nn.Module):
    def __init__(self, bert_encoder: nn.Module, enc_hid_dim=768, outputs=2, dropout=0.1):
        super().__init__()
        self.bert_encoder = bert_encoder
        self.enc_hid_dim = enc_hid_dim

    def forward(self, src, mask):
        bert_output = self.bert_encoder(src, mask)
        # finish later

def init_classification_head_weights(m: nn.Module, hidden_size=768):
    # finish later
    return

#define hyperparameters
BATCH_SIZE = 10
LR = 1e-5
WEIGHT_DECAY = 0
N_EPOCHS = 3
CLIP = 1.0

# Define models, move to device, and initialize weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model_name = 'distilbert-base-uncased' 
configuration = DistilBertConfig()
bert_encoder = DistilBertModel(configuration)
tokenizer = DistilBertTokenizer.from_pretrained("data generation/data.csv")
model = EmailClassifier(bert_encoder).to(device)
model.apply(init_classification_head_weights)
model.to(device)

# Create pytorch dataloaders
train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE,collate_fn=partial(transformer_collate_fn, tokenizer=tokenizer), shuffle = True)
val_dataloader = DataLoader(val_dataset,batch_size=BATCH_SIZE,collate_fn=partial(transformer_collate_fn, tokenizer=tokenizer))
test_dataloader = DataLoader(test_dataset,batch_size=BATCH_SIZE,collate_fn=partial(transformer_collate_fn, tokenizer=tokenizer))

# Train model
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=N_EPOCHS*len(train_dataloader))
print(f'The model has {count_parameters(model):,} trainable parameters')
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

test_loss = evaluate(model, test_dataloader, device)
test_acc = evaluate_acc(model, test_dataloader, device)
print(f'Test Loss: {test_loss:.3f}')
print(f'Test Acc: {test_acc:.3f}')