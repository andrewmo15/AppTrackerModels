import re
from functools import partial
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer

class EmailDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]

class EmailClassifier(nn.Module):
    def __init__(self, bert_encoder: nn.Module, enc_hid_dim=768, outputs=3, dropout=0.1):
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

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model

def getPrediction(email, model, device):
    model.eval()
    df = pd.DataFrame([[email, -1]], columns=['email', 'status'])
    email_dataset = EmailDataset(df)
    email_dataloader = DataLoader(email_dataset,batch_size=1,collate_fn=partial(transformer_collate_fn, tokenizer=tokenizer))
    prediction = 0
    with torch.no_grad():
        for sentences, labels, masks in email_dataloader:
            output = model(sentences.to(device), masks.to(device))
            output = F.softmax(output, dim=1)
            prediction = torch.argmax(output, dim=1)
    if prediction == 0:
        return 'SUBMITTED'
    elif prediction == 1:
        return 'REJECTED'
    else:
        return 'IRRELEVANT'

def cleanText(text):
    return re.sub(r'[^A-Za-z0-9 ]+', '', text)

model = load_checkpoint('/kaggle/input/email-model-checkpoint/checkpoint.pth')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

email = """Thank you for applying to Neuralink Hi Test, Thanks for applying to Neuralink. 
Your application has been received and we will review it right away. 
If your application seems like a good fit for the position, we will contact you soon. 
Regards, Neuralink"""
predict = getPrediction(cleanText(email), model, device)
print(predict)
email = """Hello, 
I hope this message finds you well! I wanted to thank you for your interest in Affirm 
and giving us the opportunity to consider you for the Software Engineer, New Grad role. 
Unfortunately, we are moving forward with other candidates at this point in time. 
We know there are many options out there, and want you to know we genuinely value your 
interest in working with us. That said, we would love to stay in touch as we grow. 
Feel free to connect via LinkedIn, as our company is constantly evolving and there 
may be a position available for you down the line. Wishing you the best of luck on 
your job search. Sincerely, Affirm"""
predict = getPrediction(cleanText(email), model, device)
print(predict)