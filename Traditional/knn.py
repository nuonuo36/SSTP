import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

X = np.load('tfidf_embeddings.npy')            
y = pd.read_csv('gemini_labels.csv')['majority_vote']   
y = y.iloc[:4000]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=36)

knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, y_train)
y_predicted = knn.predict(X_test)
accuracy = accuracy_score( y_test, y_predicted)
precision = precision_score(y_test, y_predicted)
recall = recall_score(y_test, y_predicted)
f1 = f1_score(y_test, y_predicted)
roc_auc = roc_auc_score(y_test, y_predicted)
conf_matrix = confusion_matrix(y_test, y_predicted)
print(precision, recall, f1, roc_auc)