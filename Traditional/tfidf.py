import pandas as pd
import google.generativeai as genai
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

#clean data
df_unclean = pd.read_csv('reddit_anxiety.csv')  
df = df_unclean.drop(columns=['author','id'])
df = df.iloc[0:4066]
df = df.dropna(subset=['selftext'])   # Remove rows where 'selftext' is NaN
x = df['selftext'].tolist()
y = df['majority_vote']

#embedding x
vectorizer = TfidfVectorizer()
tfidf_embeddings = vectorizer.fit_transform(x)

np.save('tfidf_embeddings.npy', tfidf_embeddings.toarray())    
y.to_csv('tfidf_labels.csv', index=False)      
