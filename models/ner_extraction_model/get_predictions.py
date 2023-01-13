import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

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
    inputs = torch.tensor(tokens['input_ids']).unsqueeze(0)
    masks = torch.tensor(tokens['attention_mask']).unsqueeze(0)
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        masks = masks.cuda()
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

label_list = ['O','ORG','POS']
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
if torch.cuda.is_available():
    model = load_checkpoint('ner_checkpoint.pth')
else:
    checkpoint = torch.load('ner_checkpoint.pth', map_location=torch.device('cpu'))
    model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=len(label_list))
    model.load_state_dict(checkpoint['state_dict'])

email = """Dear Andrew ,
Thank you for applying to the 2023 University Graduate - Software Engineer position.
We have carefully reviewed your application. Unfortunately, we’ve decided to move forward with other candidates for this role. While it didn’t work out this time, we encourage you to continue visiting our career site and explore other opportunities at Adobe.
Thank you again for your interest and best wishes in your job search.
Best Regards,
Adobe Talent Acquisition Team
"""
company, position = getPredictions(model, email, tokenizer)
print(company, position)
email = """Hi Andrew,
Thank you for your interest in Zip and for applying to our open position.
We know that it takes time and energy to submit for a new role. Our recruiting team carefully reviewed your background and experience and unfortunately we won’t be moving forward with your application at this time. 
We encourage you to check our career site for roles that may be a match in the future and wish you the best of luck in your search.
Thanks again,
Zip"""
company, position = getPredictions(model, email, tokenizer)
print(company, position)
email = """Hello,
I hope this message finds you well! I wanted to thank you for your interest in Affirm and giving us the opportunity to consider you for the Software Engineer, New Grad role.
Unfortunately, we are moving forward with other candidates at this point in time.
We know there are many options out there, and want you to know we genuinely value your interest in working with us. 
That said, we would love to stay in touch as we grow. Feel free to connect via LinkedIn, as our company is constantly evolving and there may be a position available for you down the line.
Wishing you the best of luck on your job search.
Sincerely,
Affirm Talent Team"""
company, position = getPredictions(model, email, tokenizer)
print(company, position)