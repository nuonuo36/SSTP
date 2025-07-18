import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# Load data
df = pd.read_csv('reddit_anxiety.csv')
df = df.iloc[0:4000]
sentences = df['selftext'].fillna('').tolist()

# Tokenize using gensim's simple_preprocess
tokenized_sentences = [simple_preprocess(str(sent), deacc=True) for sent in sentences]

# Train Word2Vec
model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    epochs=10
)

# Save model
model.save("word2vec_model.model")

# Example usage
# print(model.wv.most_similar('anxiety', topn=5))

def get_doc_embedding(tokens):
    # Filter words that exist in Word2Vec's vocabulary
    valid_words = [word for word in tokens if word in model.wv]
    if not valid_words:
        return np.zeros(model.vector_size)  # Return zeros if no valid words
    return np.mean(model.wv[valid_words], axis=0)  # Average word vectors

# Example: Embed the first post
# doc_embedding = get_doc_embedding(tokenized_sentences[0])
# print("Document embedding shape:", doc_embedding.shape)

# Generate embeddings for all posts
doc_embeddings = [get_doc_embedding(tokens) for tokens in tokenized_sentences]

# Save as numpy array
np.save('word2vec_doc_embeddings.npy', np.array(doc_embeddings))

# Save labels (if needed)
df['majority_vote'].to_csv('labels.csv', index=False)