from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Lowercase model
model = BertModel.from_pretrained('bert-base-uncased')

df_unclean = pd.read_csv('reddit_anxiety.csv')  
df = df_unclean.drop(columns=['author','id'])
df = df.iloc[0:2000]
x = df['selftext'].tolist()
y = df['majority_vote']

for i in tqdm(range(int(len(x)))):
    try:
        inputs = tokenizer(x, return_tensors="pt", padding=True, truncation=True)
    except ValueError as e:
        x = [str(text) for text in x]
        inputs = tokenizer(x, return_tensors="pt", padding=True, truncation=True)

# Generate embeddings
with torch.no_grad():
    outputs = model(**inputs)

# Pool embeddings (mean of all tokens)
token_embeddings = outputs.last_hidden_state
sentence_embeddings = torch.mean(token_embeddings, dim=1)  # Shape: [batch_size, 768]

# Save
np.save('bert_embeddings.npy', sentence_embeddings.numpy())
y.to_csv('bert_labels.csv', index=False)      