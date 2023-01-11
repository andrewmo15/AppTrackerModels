import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch

# Defining torch dataset class for email dataset
class EmailDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]

# Create collate function for email dataset that will tokenize the input emails for use with our BERT models.
def transformer_collate_fn(batch, tokenizer):
    bert_vocab = tokenizer.get_vocab()
    bert_pad_token = bert_vocab['[PAD]']
    sentences, labels, masks = [], [], []
    for data in batch:
        tokenizer_output = tokenizer([data['email']], truncation=True, max_length=512)
        tokenized_sent = tokenizer_output['input_ids'][0]
        mask = tokenizer_output['attention_mask'][0]
        sentences.append(torch.tensor(tokenized_sent))
        labels.append(torch.tensor(data['status']))
        masks.append(torch.tensor(mask))
    sentences = pad_sequence(sentences, batch_first=True, padding_value=bert_pad_token)
    labels = torch.stack(labels, dim=0)
    masks = pad_sequence(masks, batch_first=True, padding_value=0.0)
    return sentences, labels, masks

# Computes the amount of time that a training epoch took and displays it in human readable form
def epoch_time(start_time: int, end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# Train a given model, using a pytorch dataloader, optimizer, and scheduler (if provided)
def train(model, dataloader, optimizer, device, clip: float, scheduler = None):
    model.train()
    epoch_loss = 0
    for sentences, labels, masks in dataloader:
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

# Calculate the loss from the model on the provided dataloader
def evaluate(model, dataloader, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for sentences, labels, masks in dataloader:
            output = model(sentences.to(device), masks.to(device))
            loss = F.cross_entropy(output, labels.to(device))
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

# Calculate the prediction accuracy on the provided dataloader
def evaluate_acc(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total = 0
        for _, (sentences, labels, masks) in enumerate(dataloader):
            output = model(sentences.to(device), masks.to(device))
            output = F.softmax(output, dim=1)
            output_class = torch.argmax(output, dim=1)
            total_correct += torch.sum(torch.where(output_class == labels.to(device), 1, 0))
            total += sentences.size()[0]
    return total_correct / total

class EmailClassifier(nn.Module):
    def __init__(self, bert_encoder: nn.Module, outputs=3, dropout=0.1):
        super().__init__()
        self.bert_encoder = bert_encoder
        self.classifier = nn.Linear(self.bert_encoder.config.hidden_size, outputs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, mask):
        bert_output = self.bert_encoder(src, mask)
        last_hidden_layer = bert_output[0][:,-1,:]
        last_hidden_layer = self.dropout(last_hidden_layer)
        logits = self.classifier(last_hidden_layer)
        return logits