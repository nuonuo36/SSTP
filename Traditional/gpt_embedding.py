import openai
import pandas as pd
import google.generativeai as genai
import numpy as np
from tqdm import tqdm

df_unclean = pd.read_csv('reddit_anxiety.csv')  
df = df_unclean.drop(columns=['author','id'])
df = df.iloc[0:4000]
x = df['selftext'].tolist()
y = df['majority_vote']

# Set your API key
openai.api_key = ""  # Replace with your actual key

def get_embeddings(texts, model="text-embedding-3-large"):
    """Get embeddings for a list of texts."""
    response = openai.Embedding.create(
        input=texts,
        model=model)
    return response

embeddings = []
for post_i in tqdm(x, desc="Progress"):  # Add progress bar
    embeddings.append(get_embeddings(str(post_i)))


np.save('gpt_embeddings.npy', embeddings)    
y.to_csv('gpt_labels.csv', index=False)      