import pandas as pd
import torch
from tqdm.notebook import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split

# read in data
df = pd.read_csv("data generation/data.csv")

# drop irrelevant features
irrelevant_features = ["company"]
df.drop(irrelevant_features, inplace=True,axis=1)

possible_labels = df.status.unique()

label_dict = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index
df['label'] = df.status.replace(label_dict)

X_train, X_val, y_train, y_val = train_test_split(df.index.values, df.label.values, test_size=0.15, random_state=42, stratify=df.label.values)

df['data_type'] = ['not_set']*df.shape[0]

df.loc[X_train, 'data_type'] = 'train'
df.loc[X_val, 'data_type'] = 'val'

print(df.groupby(['status', 'label', 'data_type']).count())