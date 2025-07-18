import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

X = np.load('gemini_embeddings.npy')            
y = pd.read_csv('labels.csv')['majority_vote']   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=36)

gnb = GaussianNB()
y_predicted = gnb.fit(X_train, y_train).predict(X_test)

accuracy = accuracy_score( y_test, y_predicted)
precision = precision_score(y_test, y_predicted)
recall = recall_score(y_test, y_predicted)
f1 = f1_score(y_test, y_predicted)
roc_auc = roc_auc_score(y_test, y_predicted)
conf_matrix = confusion_matrix(y_test, y_predicted)
print(conf_matrix)