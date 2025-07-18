import pandas as pd
import google.generativeai as genai
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer

#clean data
df_unclean = pd.read_csv('reddit_anxiety.csv')  
df = df_unclean.drop(columns=['author','id'])
df = df.iloc[0:2000]
df = df.dropna(subset=['selftext'])  # Remove rows where 'selftext' is NaN
x = df['selftext'].tolist()
y = df['majority_vote']

#count vectorizer embedding x
vectorizer = CountVectorizer()
count_embeddings = vectorizer.fit_transform(x)

np.save('count_embeddings.npy', count_embeddings.toarray())    
y.to_csv('count_labels.csv', index=False)      