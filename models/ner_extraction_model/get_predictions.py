import torch
from transformers import AutoTokenizer

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model

def getPredictions(model, paragraph, tokenizer):
    label_list = ['O','ORG','POS']
    tokens = tokenizer(paragraph)
    inputs = torch.tensor(tokens['input_ids']).unsqueeze(0).cuda()
    masks = torch.tensor(tokens['attention_mask']).unsqueeze(0).cuda()
    predictions = model.forward(input_ids=inputs, attention_mask=masks)
    predictions = torch.argmax(predictions.logits.squeeze(), axis=1)
    predictions = [label_list[i] for i in predictions]
    words = tokenizer.batch_decode(tokens['input_ids'])
    company = getValue(predictions, words, "ORG")
    position = getValue(predictions, words, "POS")
    return company, position

def getValue(predictions, words, type):
    values = {}
    temp = ""
    flag = False
    for i, word in enumerate(words):
        if flag and predictions[i] == type:
            if "##" in word:
                temp += word[2:]
            else:
                temp += " " + word
        elif predictions[i] == type:
            flag = True
            temp += word
        elif flag and not predictions[i] == type:
            try:
                values[temp] += 1
            except KeyError:
                values[temp] = 1
            temp = ""
            flag = False
    temp = 0
    rtn = ""
    for key, count in values.items():
        if count > temp:
            rtn = key
    return rtn

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = load_checkpoint("/kaggle/working/checkpoint.pth")