import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

X = np.load('gemini_embeddings.npy')            
y = pd.read_csv('labels.csv')['majority_vote']   

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=36)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'binary:logistic',  # for binary classification
    # 'objective': 'multi:softmax',  # for multiclass
    # 'num_class': 10,               # for multiclass
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'logloss',       # or 'mlogloss' for multiclass
    'seed': 42
}
num_rounds = 100
model = xgb.train(params, dtrain, num_rounds)

y_predicted = model.predict(dtest)
y_predicted = (y_predicted > 0.5).astype(int)   # Convert to 0/1 using 0.5 threshold

accuracy = accuracy_score(y_test, y_predicted)
precision = precision_score(y_test, y_predicted)
recall = recall_score(y_test, y_predicted)
f1 = f1_score(y_test, y_predicted)
roc_auc = roc_auc_score(y_test, y_predicted)
conf_matrix = confusion_matrix(y_test, y_predicted)
print(conf_matrix)
