import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np



fake_raw = pd.read_csv('./dataset/Fake.csv')
true_raw = pd.read_csv('./dataset/True.csv')

fake_raw['is_fake'] = 1
true_raw['is_fake'] = 0
concat = pd.concat([fake_raw, true_raw])

# shuffle data
dataset = concat.sample(frac=1).reset_index(drop=True)

data_array = np.array(dataset['text'])
target_array = np.array(dataset['is_fake'])

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(data_array, target_array, test_size = 0.20, random_state = 42)



device = torch.device('cpu')

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)

tokenized_train = tokenizer(list(X_train)[:10], padding = True, truncation = True, return_tensors="pt")
tokenized_val = tokenizer(list(X_val) , padding = True, truncation = True,  return_tensors="pt")

tokenized_train = {k:torch.tensor(v).to(device) for k,v in tokenized_train.items()}
tokenized_val = {k:torch.tensor(v).to(device) for k,v in tokenized_val.items()}

@torch.no_grad()
def get_embedding(input_batch):
    data = model(**input_batch) #dim : [batch_size(nr_sentences), tokens, emb_dim]
    new_data = data.last_hidden_state[:,-1,:] + data.last_hidden_state[:,-2,:] + data.last_hidden_state[:,-3,:] + data.last_hidden_state[:,-4,:]
    return new_data

data = get_embedding(tokenized_train)
print(data)
    

# class BertBasedClassificationModel(torch.nn.Module):

#     @torch.no_grad()
#     def get_embedding():
#         hidden_train = model(**tokenized_train) #dim : [batch_size(nr_sentences), tokens, emb_dim]
#         hidden_val = model(**tokenized_val)
    

# #get only the [CLS] hidden states
# cls_train = hidden_train.last_hidden_state[:,0,:]
# cls_val = hidden_val.last_hidden_state[:,0,:]


print()


# import torch
# from transformers import AutoTokenizer, AutoModel

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)