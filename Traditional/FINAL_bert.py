import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from functions import *
import gc

# Disable tokenizer parallelism to avoid conflicts
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 1. Load and prepare data
df = pd.read_csv('reddit_anxiety.csv')
df['clean_text'] = df['selftext'].fillna('')
df['clean_text'] = df.apply(
    lambda x: x['title'] if x['clean_text'] == '' else x['clean_text'],
    axis=1
)
texts = df['clean_text']
labels = df['majority_vote']

# 2. Split and balance data
train_texts = texts.iloc[:4000]
train_labels = labels.iloc[:4000]
test_texts = texts.iloc[4000:]
test_labels = labels.iloc[4000:]

X_train, y_train = balance_data(train_texts, train_labels, 2000)
X_test, y_test = balance_data(test_texts, test_labels, 500)

# 3. Optimized BERT Embedding Generation
def get_bert_embeddings(texts, batch_size=16):
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model = model.to('cpu')  # Use CPU to avoid GPU memory issues
    model.eval()
    
    embeddings = []
    
    # Process in smaller batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating BERT embeddings"):
        batch = texts[i:i+batch_size].tolist()
        batch = [str(text) for text in batch]
        
        try:
            # Tokenize with truncation
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to('cpu')  # Keep on CPU
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Mean pooling
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(batch_embeddings)
            
            # Clean up
            del inputs, outputs, batch_embeddings
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            
        except Exception as e:
            print(f"Error processing batch {i//batch_size}: {str(e)}")
            continue
    
    # Clean up models
    del model, tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    return np.vstack(embeddings)

# Generate embeddings with smaller batches and memory management
try:
    X_train_bert = get_bert_embeddings(X_train, batch_size=8)  # Smaller batch for stability
    X_test_bert = get_bert_embeddings(X_test, batch_size=8)
except Exception as e:
    print(f"Failed to generate embeddings: {str(e)}")
    exit()

# 4. Define models (removed GaussianNB as it's not suitable for BERT embeddings)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=50),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='linear', probability=True),
    "Naive Bayes": GaussianNB(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(50,), max_iter=500)
}

# 5. Train and evaluate
results = []
for name, model in models.items():
    try:
        model.fit(X_train_bert, y_train)
        y_pred = model.predict(X_test_bert)
        y_proba = model.predict_proba(X_test_bert)[:, 1] if hasattr(model, "predict_proba") else [0]*len(y_test)
        
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, y_proba) if len(set(y_test)) > 1 else float('nan')
        })
    except Exception as e:
        print(f"Error with {name}: {str(e)}")
        continue

# 6. Save results
results_df = pd.DataFrame(results)
results_df.to_excel("FINAL_Bert_Embedding.xlsx")
print(results_df)
