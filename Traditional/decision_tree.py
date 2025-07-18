import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

X = np.load('bert_embeddings.npy')            
y = pd.read_csv('bert_labels.csv')['majority_vote']   

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=36)

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
y_predicted = tree.predict(X_test)

accuracy = accuracy_score(y_test, y_predicted)
precision = precision_score(y_test, y_predicted)
recall = recall_score(y_test, y_predicted)
f1 = f1_score(y_test, y_predicted)
roc_auc = roc_auc_score(y_test, y_predicted)
conf_matrix = confusion_matrix(y_test, y_predicted)
print(conf_matrix)