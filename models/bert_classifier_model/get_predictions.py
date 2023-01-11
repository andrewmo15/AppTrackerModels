import re
from functools import partial
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from model_utils import EmailDataset, EmailClassifier, transformer_collate_fn

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
        for sentences, _, masks in email_dataloader:
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

bert_model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(bert_model_name)
if torch.cuda.is_available():
    model = load_checkpoint('classifier_checkpoint.pth')
    device = torch.device('cuda')
else:
    bert_model = DistilBertModel.from_pretrained(bert_model_name)
    model = EmailClassifier(bert_model)
    checkpoint = torch.load('classifier_checkpoint.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    device = torch.device('cpu')

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