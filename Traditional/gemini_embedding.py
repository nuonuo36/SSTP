import pandas as pd
import google.generativeai as genai
import numpy as np
from tqdm import tqdm

#clean data
df_unclean = pd.read_csv('reddit_anxiety.csv')  
df = df_unclean.drop(columns=['author','id'])
df = df.iloc[0:4000]
x = df['selftext'].tolist()
y = df['majority_vote']

#gemini embedding x
genai.configure(api_key="AIzaSyBpgrGwUVhU9tTZNkRQ5RyMVBYlDRe5TAI") 
def embedding(one_post):
    model="models/embedding-001"
    return genai.embed_content(model=model, content=one_post)['embedding']

embeddings = []
for post_i in tqdm(x, desc="Progress"):  # Add progress bar
    embeddings.append(embedding(str(post_i)))

np.save('gemini_embeddings.npy', embeddings)    
y.to_csv('gemini_labels.csv', index=False)      