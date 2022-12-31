import re
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForTokenClassification, logging
import pandas as pd
import numpy as np
from datasets import Dataset, load_metric
logging.set_verbosity_error()

# NER tags replaced already {0: OTHER, 1: ORG, 2: POS}
def helper(row):
    text = row["email"]
    company = row["company"]
    position = row["position"]
    conditions = 'A-Za-z0-9 '
    for char in company + position:
        if not "A" <= char <= "Z" and not "a" <= char <= "z" and not "0" <= char <= "9" and not char in conditions:
            conditions += char
    conditions = '[^' + conditions + ']+'
    text = re.sub(r'{}'.format(conditions), '', text)
    text = text.split()
    company = company.split()
    position = position.split()
    tags = []
    i = 0
    while i <= len(text) - min(len(company), len(position)):
        if i + len(company) <= len(text) and text[i: i + len(company)] == company:
            tags += ["ORG" for _ in range(len(company))]
            i += len(company)
        elif i + len(position) <= len(text) and text[i: i + len(position)] == position:
            tags += ["POS" for _ in range(len(position))]
            i += len(position)
        else:
            tags.append("O")
            i += 1
    while not len(tags) == len(text):
        tags.append("O")
    texttext = []
    tagstags = []
    for i in range(len(text)):
        a = re.sub(r'[^A-Za-z0-9 ]+', '', text[i])
        if tags[i] == "O" and a == "":
            continue
        else:
            texttext.append(text[i])
            tagstags.append(tags[i])
    return texttext, tagstags

# Read in data
df = pd.read_csv("/kaggle/input/neremaildata/data.csv")

# Create text column of combined data and remove all non alphanumeric characters
df["email"] = df["from"] + " " + df["subject"] + " " + df["body"]
df["token+tags"] = df[["position", "company", "email"]].apply(helper, axis =1)
df["tokens"] = df["token+tags"].apply(lambda x: x[0])
df["ner_tags"] = df["token+tags"].apply(lambda x: x[1])

# Drop irrelevant features
irrelevant_features = ["status", "from", "subject", "body", "position", "company", "email", "token+tags"]
df.drop(irrelevant_features, inplace=True,axis=1)

# Divide data into train, validation, and test datasets
train_ratio = 0.75
validation_ratio = 0.15
test_ratio = 0.10

train_data, test_data, y_train, y_test = train_test_split(df[["tokens"]], df[['ner_tags']], test_size=1 - train_ratio)
val_data, test_data, y_val, y_test = train_test_split(test_data, y_test, test_size=test_ratio/(test_ratio + validation_ratio))

temp = y_train["ner_tags"].values.tolist()
train_data["ner_tags"] = temp
temp = y_test["ner_tags"].values.tolist()
test_data["ner_tags"] = temp
# ====================================
label_list = ['O','ORG','POS']
label_encoding_dict = {'O': 0, 'ORG': 1, 'POS': 2}

train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)

task = "ner" 
model_checkpoint = "distilbert-base-uncased"
batch_size = 16
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_and_align_labels(examples):
    label_all_tokens = True
    tokenized_inputs = tokenizer(list(examples["tokens"]), truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif label[word_idx] == '0':
                label_ids.append(0)
            elif word_idx != previous_word_idx:
                label_ids.append(label_encoding_dict[label[word_idx]])
            else:
                label_ids.append(label_encoding_dict[label[word_idx]] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
        
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


train_tokenized_datasets = train_dataset.map(tokenize_and_align_labels, batched=True)
test_tokenized_datasets = test_dataset.map(tokenize_and_align_labels, batched=True)

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))

args = TrainingArguments(
    f"test-{task}",
    evaluation_strategy = "epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=1e-5,
)

data_collator = DataCollatorForTokenClassification(tokenizer)
metric = load_metric("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}
    
trainer = Trainer(
    model,
    args,
    train_dataset=train_tokenized_datasets,
    eval_dataset=test_tokenized_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()