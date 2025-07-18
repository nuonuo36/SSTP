import pandas as pd
import numpy as np
from functions import *
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Load the data
df_unsorted = pd.read_csv('reddit_anxiety.csv')
df_unsorted['clean_text'] = df_unsorted['selftext'].fillna('')
df_unsorted['clean_text'] = df_unsorted.apply(
    lambda x: x['title'] if x['clean_text'] == '' else x['clean_text'],
    axis=1)
texts = df_unsorted['clean_text']
labels = df_unsorted['majority_vote']

# Balance training set (2000 each)
X_train, y_train = balance_data(texts, labels, 2000)
X_train = X_train.reset_index(drop=True)  # Remove duplicate indices
y_train = y_train.reset_index(drop=True) 
df = pd.DataFrame({
    'selftext': X_train,
    'majority_vote': y_train
})

df = df.iloc[:4000]
df.to_excel("SDC.xlsx")

embeddings = np.load("tfidf_embeddings.npy")  

# Normalize embedding values
embeddings = (embeddings - embeddings.mean(0)) / embeddings.std(0)

# Combine embeddings for all positive samples
# positive_case_indices = df[df['majority_vote'] == 1].index[:20].tolist()
positive_case_indices = [0,36,43,52,57,79,94,152,181,232,250,262,278,292,342,381,425,554,591,601,604,624,814,900,929,938,1003,1015,1286,1295,1349,1462,1509,1556,1579,1754,1811,1850,1866,1908,1961,2009,2054,2275,2320,2340,2379,2401,2519,2538,2577,2587,2824,2878,3005,3083,3101,3152,3185,3194,3255,3291,3356,3364,3381,3408,3411,3448,3453,3503,3648,3802,3843]
# 3527,3582, 2038,2046,16,19,20, 3756,3921,3998
signature = embeddings[positive_case_indices].mean(axis=0)
# Multiply each sample's dimension values by the signature, sum to a single value per sample
df["Score"] = embeddings @ signature

df = df.sort_values("Score", ascending=False)

y_pred = 0
for i in range(0,2000):
    if df.iloc[i]["majority_vote"] == 1:
        y_pred += 1
accuracy = y_pred / 2000
print(y_pred)
print(accuracy)





#just do it manually; tfidf doesnt work cuz count and tfidf works differently. With the right embeddings, SDC gets quite close to the fine-tuned transformers.
# count_df = pd.read_excel('FINAL_Count_Vectorizer.xlsx', index_col='Model')
# count_df['SDC_Accuracy'] = sdc_results['Accuracy']
# count_df.to_excel('FINAL_Count_Vectorizer.xlsx')

# x_test = texts.iloc[4000:]
# X_train_counts = df.iloc[:3000]['Score'].values.reshape(-1, 1)
# X_test_counts = df.iloc[3000:]['Score'].values.reshape(-1, 1)
# y_train = y_train[:3000]
# y_test = y_test = labels.iloc[3000:4000]
# # 4. Define models to evaluate
# models = {
#     "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
#     "Decision Tree": DecisionTreeClassifier(max_depth=5),
#     "Random Forest": RandomForestClassifier(n_estimators=100),
#     "Gradient Boosting": GradientBoostingClassifier(n_estimators=50),
#     "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
#     "SVM": SVC(kernel='linear', probability=True),
#     "Naive Bayes": GaussianNB(),
#     "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
#     "Neural Network": MLPClassifier(hidden_layer_sizes=(50,), max_iter=500)
# }

# # 5. Train and evaluate all models
# results = []
# for name, model in models.items():
#     # Train
#     model.fit(X_train_counts, y_train)
    
#     # Predict
#     y_pred = model.predict(X_test_counts)
#     y_proba = model.predict_proba(X_test_counts)[:, 1] if hasattr(model, "predict_proba") else [0]*len(y_test)
#     # Evaluate
#     results.append({
#         'Model': name,
#         'Accuracy': accuracy_score(y_test, y_pred),
#         'Precision': precision_score(y_test, y_pred),
#         'Recall': recall_score(y_test, y_pred),
#         'F1': f1_score(y_test, y_pred),
#         'ROC AUC': roc_auc_score(y_test, y_proba) if len(set(y_test)) > 1 else float('nan')
#         #  'conf_matrix' : confusion_matrix(y_test, y_proba)
#     })

# # 6. Display results
# results_df = pd.DataFrame(results)
# results_df.to_excel("FINAL_Count_SDC_Embedding.xlsx")

# # # 7. Optional: Save best model
# # best_model = models[results_df.iloc[0]['Model']]
# # # ... (save using pickle/joblib)
# # cnt = 0
# # for i in y_pred:
# #     if i == 1:
# #         cnt +=1
# # print(cnt)

