import pandas as pd
import torch
from transformers import AutoTokenizer
import numpy as np
from sklearn.model_selection import train_test_split
from model import BertBasedClassificationModel
from config import n_epoches


fake_raw = pd.read_csv('./dataset/Fake.csv')
true_raw = pd.read_csv('./dataset/True.csv')

fake_raw['is_fake'] = 1
true_raw['is_fake'] = 0
concat = pd.concat([fake_raw, true_raw])

# shuffle data
dataset = concat.sample(frac=1).reset_index(drop=True)

data_array = np.array(dataset['text'])
target_array = np.array(dataset['is_fake'])

X_train, X_val, y_train, y_val = train_test_split(data_array, target_array, test_size = 0.20, random_state = 42)
X_train = X_train[:2]
y_train = torch.tensor(y_train[:2], dtype=torch.long)


device = torch.device('cpu')

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

tokenized_train = tokenizer(list(X_train), padding = True, truncation = True, return_tensors="pt")
tokenized_val = tokenizer(list(X_val) , padding = True, truncation = True,  return_tensors="pt")

tokenized_train = {k:torch.tensor(v).to(device) for k,v in tokenized_train.items()}
tokenized_val = {k:torch.tensor(v).to(device) for k,v in tokenized_val.items()}




model = BertBasedClassificationModel(device)
criterion = torch.nn.CrossEntropyLoss()


for epoch in range(1, n_epoches+1):
    predict = model.forward(tokenized_train)
    loss = criterion(predict, y_train)
    loss.backward()
    print(predict, y_train)

data = model.infer(tokenized_train)
print(data, y_train)  


print()