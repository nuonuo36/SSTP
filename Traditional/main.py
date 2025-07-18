from functions import *

size = 5000
embeddings = ['BERT', 'Count Vectorizer', 'Gemini', 'TF-IDF', 'Word2Vec']
models = [
    'Decision Tree', 'Gradient Boosting', 'KNN', 
    'Logistic Regression', 'XGBoost', 'Naive Bayes',
    'Random Forest', 'SVM', 'Neural Network'
    # , 'Scaled Dimension Combination'
]
evaluations = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']



# bert_X_train, bert_X_test, bert_y_train, bert_y_test = bert_embedding(size)
# bert_dt_accuracy, bert_dt_precision, bert_dt_recall, bert_dt_f1, bert_dt_roc_auc, bert_dt_conf_matrix = decision_tree(bert_X_train, bert_X_test, bert_y_train, bert_y_test)
# bert_gbc_accuracy, bert_gbc_precision, bert_gbc_recall, bert_gbc_f1, bert_gbc_roc_auc, bert_gbc_conf_matrix = gradient_boosting_classifier(bert_X_train, bert_X_test, bert_y_train, bert_y_test)
# bert_knn_accuracy, bert_knn_precision, bert_knn_recall, bert_knn_f1, bert_knn_roc_auc, bert_knn_conf_matrix = knn(bert_X_train, bert_X_test, bert_y_train, bert_y_test)
# bert_lg_accuracy, bert_lg_precision, bert_lg_recall, bert_lg_f1, bert_lg_roc_auc, bert_lg_conf_matrix = Logistic_Regression(bert_X_train, bert_X_test, bert_y_train, bert_y_test)
# bert_xgb_accuracy, bert_xgb_precision, bert_xgb_recall, bert_xgb_f1, bert_xgb_roc_auc, bert_xgb_conf_matrix = XGBoost(bert_X_train, bert_X_test, bert_y_train, bert_y_test)
# bert_nb_accuracy, bert_nb_precision, bert_nb_recall, bert_nb_f1, bert_nb_roc_auc, bert_nb_conf_matrix = naive_bayes(bert_X_train, bert_X_test, bert_y_train, bert_y_test)
# bert_rf_accuracy, bert_rf_precision, bert_rf_recall, bert_rf_f1, bert_rf_roc_auc, bert_rf_conf_matrix = random_forest(bert_X_train, bert_X_test, bert_y_train, bert_y_test)
# bert_svm_accuracy, bert_svm_precision, bert_svm_recall, bert_svm_f1, bert_svm_roc_auc, bert_svm_conf_matrix = SVM(bert_X_train, bert_X_test, bert_y_train, bert_y_test)
# bert_nn_accuracy, bert_nn_precision, bert_nn_recall, bert_nn_f1, bert_nn_roc_auc, bert_nn_conf_matrix = nn(bert_X_train, bert_X_test, bert_y_train, bert_y_test)
# bert_accuracy = [bert_dt_accuracy, bert_gbc_accuracy, bert_knn_accuracy, bert_lg_accuracy, bert_xgb_accuracy, bert_nb_accuracy, bert_rf_accuracy, bert_svm_accuracy, bert_nn_accuracy]
# bert_precision = [bert_dt_precision,bert_gbc_precision,bert_knn_precision,bert_lg_precision,bert_xgb_precision,bert_nb_precision,bert_rf_precision,bert_svm_precision,bert_nn_precision]
# bert_recall = [bert_dt_recall,bert_gbc_recall,bert_knn_recall,bert_lg_recall,bert_xgb_recall,bert_nb_recall,bert_rf_recall,bert_svm_recall,bert_nn_recall]
# bert_f1 = [bert_dt_f1,bert_gbc_f1,bert_knn_f1,bert_lg_f1,bert_xgb_f1,bert_nb_f1,bert_rf_f1,bert_svm_f1,bert_nn_f1]
# bert_roc_auc = [bert_dt_roc_auc,bert_gbc_roc_auc,bert_knn_roc_auc,bert_lg_roc_auc,bert_xgb_roc_auc,bert_nb_roc_auc,bert_rf_roc_auc,bert_svm_roc_auc,bert_nn_roc_auc]
# bert_conf_matrix = [bert_dt_conf_matrix,bert_gbc_conf_matrix,bert_knn_conf_matrix,bert_lg_conf_matrix,bert_xgb_conf_matrix,bert_nb_conf_matrix,bert_rf_conf_matrix,bert_svm_conf_matrix,bert_nn_conf_matrix]

# bert_df = pd.DataFrame({
#     'Model': models,
#     'Accuracy': bert_accuracy,
#     'Precision': bert_precision,
#     'Recall': bert_recall,
#     'F1': bert_f1,
#     'AUC_ROC': bert_roc_auc,
# })
# bert_df.set_index('Model', inplace=True)
# bert_df.to_excel('bert_metrics_full.xlsx')

# bert = np.array([bert_accuracy, bert_precision, bert_recall, bert_f1, bert_roc_auc])
# bert = np.round(bert, 2)
# fig, ax = plt.subplots()
# im = ax.imshow(bert)
# ax.set_xticks(range(len(models)), labels=models,
#               rotation=45, ha="right", rotation_mode="anchor")
# ax.set_yticks(range(len(evaluations)), labels=evaluations)
# # Loop over data dimensions and create text annotations.
# for i in range(len(evaluations)):
#     for j in range(len(models)):
#         text = ax.text(j, i, bert[i, j],
#                        ha="center", va="center", color="w")
# ax.set_title("Bert Comparison")
# fig.tight_layout()
# plt.savefig('bert.png', bbox_inches='tight')
# plt.show()


# count_X_train, count_X_test, count_y_train, count_y_test = count_vectorizer(size)
# count_dt_accuracy, count_dt_precision, count_dt_recall, count_dt_f1, count_dt_roc_auc, count_dt_conf_matrix = decision_tree(count_X_train, count_X_test, count_y_train, count_y_test)
# count_gbc_accuracy, count_gbc_precision, count_gbc_recall, count_gbc_f1, count_gbc_roc_auc, count_gbc_conf_matrix = gradient_boosting_classifier(count_X_train, count_X_test, count_y_train, count_y_test)
# count_knn_accuracy, count_knn_precision, count_knn_recall, count_knn_f1, count_knn_roc_auc, count_knn_conf_matrix = knn(count_X_train, count_X_test, count_y_train, count_y_test)
# count_lg_accuracy, count_lg_precision, count_lg_recall, count_lg_f1, count_lg_roc_auc, count_lg_conf_matrix = Logistic_Regression(count_X_train, count_X_test, count_y_train, count_y_test)
# count_xgb_accuracy, count_xgb_precision, count_xgb_recall, count_xgb_f1, count_xgb_roc_auc, count_xgb_conf_matrix = XGBoost(count_X_train, count_X_test, count_y_train, count_y_test)
# count_nb_accuracy, count_nb_precision, count_nb_recall, count_nb_f1, count_nb_roc_auc, count_nb_conf_matrix = naive_bayes(count_X_train, count_X_test, count_y_train, count_y_test)
# count_rf_accuracy, count_rf_precision, count_rf_recall, count_rf_f1, count_rf_roc_auc, count_rf_conf_matrix = random_forest(count_X_train, count_X_test, count_y_train, count_y_test)
# count_svm_accuracy, count_svm_precision, count_svm_recall, count_svm_f1, count_svm_roc_auc, count_svm_conf_matrix = SVM(count_X_train, count_X_test, count_y_train, count_y_test)
# count_nn_accuracy, count_nn_precision, count_nn_recall, count_nn_f1, count_nn_roc_auc, count_nn_conf_matrix = nn(count_X_train, count_X_test, count_y_train, count_y_test)
# count_accuracy = [count_dt_accuracy,count_gbc_accuracy,count_knn_accuracy,count_lg_accuracy,count_xgb_accuracy,count_nb_accuracy,count_rf_accuracy,count_svm_accuracy,count_nn_accuracy]
# count_precision = [count_dt_precision,count_gbc_precision,count_knn_precision,count_lg_precision,count_xgb_precision,count_nb_precision,count_rf_precision,count_svm_precision,count_nn_precision]
# count_recall = [count_dt_recall,count_gbc_recall,count_knn_recall,count_lg_recall,count_xgb_recall,count_nb_recall,count_rf_recall,count_svm_recall,count_nn_recall]
# count_f1 = [count_dt_f1,count_gbc_f1,count_knn_f1,count_lg_f1,count_xgb_f1,count_nb_f1,count_rf_f1,count_svm_f1,count_nn_f1]
# count_roc_auc = [count_dt_roc_auc,count_gbc_roc_auc,count_knn_roc_auc,count_lg_roc_auc,count_xgb_roc_auc,count_nb_roc_auc,count_rf_roc_auc,count_svm_roc_auc,count_nn_roc_auc]
# count_conf_matrix = [count_dt_conf_matrix,count_gbc_conf_matrix,count_knn_conf_matrix,count_lg_conf_matrix,count_xgb_conf_matrix,count_nb_conf_matrix,count_rf_conf_matrix,count_svm_conf_matrix,count_nn_conf_matrix]

# count_df = pd.DataFrame({
#     'Model': models,
#     'Accuracy': count_accuracy,
#     'Precision': count_precision,
#     'Recall': count_recall,
#     'F1': count_f1,
#     'AUC_ROC': count_roc_auc,
# })
# count_df.set_index('Model', inplace=True)
# count_df.to_excel('count_metrics_full.xlsx')

# count = np.array([count_accuracy, count_precision, count_recall, count_f1, count_roc_auc])
# count = np.round(count, 2)
# fig, ax = plt.subplots(figsize=(12, 6))
# im = ax.imshow(count, cmap='YlGnBu', vmin=0, vmax=1)
# ax.set_xticks(range(len(models)), labels=models,
#               rotation=45, ha="right", rotation_mode="anchor")
# ax.set_yticks(range(len(evaluations)), labels=evaluations)
# for i in range(len(evaluations)):
#     for j in range(len(models)):
#         text_color = 'white' if count[i,j] > 0.5 else 'black'  
#         ax.text(j, i, count[i, j],
#                 ha="center", va="center", color=text_color)
# cbar = plt.colorbar(im)
# cbar.set_label('Score')
# ax.set_title("Count Vectorizer Model Performance Comparison")
# fig.tight_layout()
# plt.savefig('count_vectorizer_metrics.png', dpi=300, bbox_inches='tight')  
# plt.show()


# gemini_X_train, gemini_X_test, gemini_y_train, gemini_y_test = gemini_embedding(size)
# gemini_dt_accuracy, gemini_dt_precision, gemini_dt_recall, gemini_dt_f1, gemini_dt_roc_auc, gemini_dt_conf_matrix = decision_tree(gemini_X_train, gemini_X_test, gemini_y_train, gemini_y_test)
# gemini_gbc_accuracy, gemini_gbc_precision, gemini_gbc_recall, gemini_gbc_f1, gemini_gbc_roc_auc, gemini_gbc_conf_matrix = gradient_boosting_classifier(gemini_X_train, gemini_X_test, gemini_y_train, gemini_y_test)
# gemini_knn_accuracy, gemini_knn_precision, gemini_knn_recall, gemini_knn_f1, gemini_knn_roc_auc, gemini_knn_conf_matrix = knn(gemini_X_train, gemini_X_test, gemini_y_train, gemini_y_test)
# gemini_lg_accuracy, gemini_lg_precision, gemini_lg_recall, gemini_lg_f1, gemini_lg_roc_auc, gemini_lg_conf_matrix = Logistic_Regression(gemini_X_train, gemini_X_test, gemini_y_train, gemini_y_test)
# gemini_xgb_accuracy, gemini_xgb_precision, gemini_xgb_recall, gemini_xgb_f1, gemini_xgb_roc_auc, gemini_xgb_conf_matrix = XGBoost(gemini_X_train, gemini_X_test, gemini_y_train, gemini_y_test)
# gemini_nb_accuracy, gemini_nb_precision, gemini_nb_recall, gemini_nb_f1, gemini_nb_roc_auc, gemini_nb_conf_matrix = naive_bayes(gemini_X_train, gemini_X_test, gemini_y_train, gemini_y_test)
# gemini_rf_accuracy, gemini_rf_precision, gemini_rf_recall, gemini_rf_f1, gemini_rf_roc_auc, gemini_rf_conf_matrix = random_forest(gemini_X_train, gemini_X_test, gemini_y_train, gemini_y_test)
# gemini_svm_accuracy, gemini_svm_precision, gemini_svm_recall, gemini_svm_f1, gemini_svm_roc_auc, gemini_svm_conf_matrix = SVM(gemini_X_train, gemini_X_test, gemini_y_train, gemini_y_test)
# gemini_nn_accuracy, gemini_nn_precision, gemini_nn_recall, gemini_nn_f1, gemini_nn_roc_auc, gemini_nn_conf_matrix = nn(gemini_X_train, gemini_X_test, gemini_y_train, gemini_y_test)
# gemini_accuracy = [gemini_dt_accuracy,gemini_gbc_accuracy,gemini_knn_accuracy,gemini_lg_accuracy,gemini_xgb_accuracy,gemini_nb_accuracy,gemini_rf_accuracy,gemini_svm_accuracy,gemini_nn_accuracy]
# gemini_precision = [gemini_dt_precision,gemini_gbc_precision,gemini_knn_precision,gemini_lg_precision,gemini_xgb_precision,gemini_nb_precision,gemini_rf_precision,gemini_svm_precision,gemini_nn_precision]
# gemini_recall = [gemini_dt_recall,gemini_gbc_recall,gemini_knn_recall,gemini_lg_recall,gemini_xgb_recall,gemini_nb_recall,gemini_rf_recall,gemini_svm_recall,gemini_nn_recall]
# gemini_f1 = [gemini_dt_f1,gemini_gbc_f1,gemini_knn_f1,gemini_lg_f1,gemini_xgb_f1,gemini_nb_f1,gemini_rf_f1,gemini_svm_f1,gemini_nn_f1]
# gemini_roc_auc = [gemini_dt_roc_auc,gemini_gbc_roc_auc,gemini_knn_roc_auc,gemini_lg_roc_auc,gemini_xgb_roc_auc,gemini_nb_roc_auc,gemini_rf_roc_auc,gemini_svm_roc_auc,gemini_nn_roc_auc]
# gemini_conf_matrix = [gemini_dt_conf_matrix,gemini_gbc_conf_matrix,gemini_knn_conf_matrix,gemini_lg_conf_matrix,gemini_xgb_conf_matrix,gemini_nb_conf_matrix,gemini_rf_conf_matrix,gemini_svm_conf_matrix,gemini_nn_conf_matrix]

# gemini_df = pd.DataFrame({
#     'Model': models,
#     'Accuracy': gemini_accuracy,
#     'Precision': gemini_precision,
#     'Recall': gemini_recall,
#     'F1': gemini_f1,
#     'AUC_ROC': gemini_roc_auc,
# })
# gemini_df.set_index('Model', inplace=True)
# gemini_df.to_excel('gemini_metrics_full.xlsx')

# gemini = np.array([gemini_accuracy, gemini_precision, gemini_recall, gemini_f1, gemini_roc_auc])
# gemini = np.round(gemini, 2)
# fig, ax = plt.subplots(figsize=(12, 6))
# im = ax.imshow(gemini, cmap='YlGnBu', vmin=0, vmax=1)
# ax.set_xticks(range(len(models)), labels=models,
#               rotation=45, ha="right", rotation_mode="anchor")
# ax.set_yticks(range(len(evaluations)), labels=evaluations)
# for i in range(len(evaluations)):
#     for j in range(len(models)):
#         text_color = 'white' if gemini[i,j] > 0.5 else 'black'  
#         ax.text(j, i, gemini[i, j],
#                 ha="center", va="center", color=text_color)
# cbar = plt.colorbar(im)
# cbar.set_label('Score')
# ax.set_title("Gemini Model Performance Comparison")
# fig.tight_layout()
# plt.savefig('gemini_metrics.png', dpi=300, bbox_inches='tight')  
# plt.show()


# tfidf_X_train, tfidf_X_test, tfidf_y_train, tfidf_y_test = tfidf(size)
# tfidf_dt_accuracy, tfidf_dt_precision, tfidf_dt_recall, tfidf_dt_f1, tfidf_dt_roc_auc, tfidf_dt_conf_matrix = decision_tree(tfidf_X_train, tfidf_X_test, tfidf_y_train, tfidf_y_test)
# tfidf_gbc_accuracy, tfidf_gbc_precision, tfidf_gbc_recall, tfidf_gbc_f1, tfidf_gbc_roc_auc, tfidf_gbc_conf_matrix = gradient_boosting_classifier(tfidf_X_train, tfidf_X_test, tfidf_y_train, tfidf_y_test)
# tfidf_knn_accuracy, tfidf_knn_precision, tfidf_knn_recall, tfidf_knn_f1, tfidf_knn_roc_auc, tfidf_knn_conf_matrix = knn(tfidf_X_train, tfidf_X_test, tfidf_y_train, tfidf_y_test)
# tfidf_lg_accuracy, tfidf_lg_precision, tfidf_lg_recall, tfidf_lg_f1, tfidf_lg_roc_auc, tfidf_lg_conf_matrix = Logistic_Regression(tfidf_X_train, tfidf_X_test, tfidf_y_train, tfidf_y_test)
# tfidf_xgb_accuracy, tfidf_xgb_precision, tfidf_xgb_recall, tfidf_xgb_f1, tfidf_xgb_roc_auc, tfidf_xgb_conf_matrix = XGBoost(tfidf_X_train, tfidf_X_test, tfidf_y_train, tfidf_y_test)
# tfidf_nb_accuracy, tfidf_nb_precision, tfidf_nb_recall, tfidf_nb_f1, tfidf_nb_roc_auc, tfidf_nb_conf_matrix = naive_bayes(tfidf_X_train, tfidf_X_test, tfidf_y_train, tfidf_y_test)
# tfidf_rf_accuracy, tfidf_rf_precision, tfidf_rf_recall, tfidf_rf_f1, tfidf_rf_roc_auc, tfidf_rf_conf_matrix = random_forest(tfidf_X_train, tfidf_X_test, tfidf_y_train, tfidf_y_test)
# tfidf_svm_accuracy, tfidf_svm_precision, tfidf_svm_recall, tfidf_svm_f1, tfidf_svm_roc_auc, tfidf_svm_conf_matrix = SVM(tfidf_X_train, tfidf_X_test, tfidf_y_train, tfidf_y_test)
# tfidf_nn_accuracy, tfidf_nn_precision, tfidf_nn_recall, tfidf_nn_f1, tfidf_nn_roc_auc, tfidf_nn_conf_matrix = nn(tfidf_X_train, tfidf_X_test, tfidf_y_train, tfidf_y_test)
# tfidf_accuracy = [tfidf_dt_accuracy,tfidf_gbc_accuracy,tfidf_knn_accuracy,tfidf_lg_accuracy,tfidf_xgb_accuracy,tfidf_nb_accuracy,tfidf_rf_accuracy,tfidf_svm_accuracy,tfidf_nn_accuracy]
# tfidf_precision = [tfidf_dt_precision,tfidf_gbc_precision,tfidf_knn_precision,tfidf_lg_precision,tfidf_xgb_precision,tfidf_nb_precision,tfidf_rf_precision,tfidf_svm_precision,tfidf_nn_precision]
# tfidf_recall = [tfidf_dt_recall,tfidf_gbc_recall,tfidf_knn_recall,tfidf_lg_recall,tfidf_xgb_recall,tfidf_nb_recall,tfidf_rf_recall,tfidf_svm_recall,tfidf_nn_recall]
# tfidf_f1 = [tfidf_dt_f1,tfidf_gbc_f1,tfidf_knn_f1,tfidf_lg_f1,tfidf_xgb_f1,tfidf_nb_f1,tfidf_rf_f1,tfidf_svm_f1,tfidf_nn_f1]
# tfidf_roc_auc = [tfidf_dt_roc_auc,tfidf_gbc_roc_auc,tfidf_knn_roc_auc,tfidf_lg_roc_auc,tfidf_xgb_roc_auc,tfidf_nb_roc_auc,tfidf_rf_roc_auc,tfidf_svm_roc_auc,tfidf_nn_roc_auc]
# tfidf_conf_matrix = [tfidf_dt_conf_matrix,tfidf_gbc_conf_matrix,tfidf_knn_conf_matrix,tfidf_lg_conf_matrix,tfidf_xgb_conf_matrix,tfidf_nb_conf_matrix,tfidf_rf_conf_matrix,tfidf_svm_conf_matrix,tfidf_nn_conf_matrix]

# tfidf_df = pd.DataFrame({
#     'Model': models,
#     'Accuracy': tfidf_accuracy,
#     'Precision': tfidf_precision,
#     'Recall': tfidf_recall,
#     'F1': tfidf_f1,
#     'AUC_ROC': tfidf_roc_auc,
# })
# tfidf_df.set_index('Model', inplace=True)
# tfidf_df.to_excel('tfidf_metrics_full.xlsx')

# tfidf = np.array([tfidf_accuracy, tfidf_precision, tfidf_recall, tfidf_f1, tfidf_roc_auc])
# tfidf = np.round(tfidf, 2)
# fig, ax = plt.subplots(figsize=(12, 6))
# im = ax.imshow(tfidf, cmap='YlGnBu', vmin=0, vmax=1)
# ax.set_xticks(range(len(models)), labels=models,
#               rotation=45, ha="right", rotation_mode="anchor")
# ax.set_yticks(range(len(evaluations)), labels=evaluations)
# for i in range(len(evaluations)):
#     for j in range(len(models)):
#         text_color = 'white' if tfidf[i,j] > 0.5 else 'black'  
#         ax.text(j, i, tfidf[i, j],
#                 ha="center", va="center", color=text_color)
# cbar = plt.colorbar(im)
# cbar.set_label('Score')
# ax.set_title("TF-IDF Model Performance Comparison")
# fig.tight_layout()
# plt.savefig('tfidf_metrics.png', dpi=300, bbox_inches='tight')  
# plt.show()


# word2vec_X_train, word2vec_X_test, word2vec_y_train, word2vec_y_test = word2vec(size)
# word2vec_dt_accuracy, word2vec_dt_precision, word2vec_dt_recall, word2vec_dt_f1, word2vec_dt_roc_auc, word2vec_dt_conf_matrix = decision_tree(word2vec_X_train, word2vec_X_test, word2vec_y_train, word2vec_y_test)
# word2vec_gbc_accuracy, word2vec_gbc_precision, word2vec_gbc_recall, word2vec_gbc_f1, word2vec_gbc_roc_auc, word2vec_gbc_conf_matrix = gradient_boosting_classifier(word2vec_X_train, word2vec_X_test, word2vec_y_train, word2vec_y_test)
# word2vec_knn_accuracy, word2vec_knn_precision, word2vec_knn_recall, word2vec_knn_f1, word2vec_knn_roc_auc, word2vec_knn_conf_matrix = knn(word2vec_X_train, word2vec_X_test, word2vec_y_train, word2vec_y_test)
# word2vec_lg_accuracy, word2vec_lg_precision, word2vec_lg_recall, word2vec_lg_f1, word2vec_lg_roc_auc, word2vec_lg_conf_matrix = Logistic_Regression(word2vec_X_train, word2vec_X_test, word2vec_y_train, word2vec_y_test)
# word2vec_xgb_accuracy, word2vec_xgb_precision, word2vec_xgb_recall, word2vec_xgb_f1, word2vec_xgb_roc_auc, word2vec_xgb_conf_matrix = XGBoost(word2vec_X_train, word2vec_X_test, word2vec_y_train, word2vec_y_test)
# word2vec_nb_accuracy, word2vec_nb_precision, word2vec_nb_recall, word2vec_nb_f1, word2vec_nb_roc_auc, word2vec_nb_conf_matrix = naive_bayes(word2vec_X_train, word2vec_X_test, word2vec_y_train, word2vec_y_test)
# word2vec_rf_accuracy, word2vec_rf_precision, word2vec_rf_recall, word2vec_rf_f1, word2vec_rf_roc_auc, word2vec_rf_conf_matrix = random_forest(word2vec_X_train, word2vec_X_test, word2vec_y_train, word2vec_y_test)
# word2vec_svm_accuracy, word2vec_svm_precision, word2vec_svm_recall, word2vec_svm_f1, word2vec_svm_roc_auc, word2vec_svm_conf_matrix = SVM(word2vec_X_train, word2vec_X_test, word2vec_y_train, word2vec_y_test)
# word2vec_nn_accuracy, word2vec_nn_precision, word2vec_nn_recall, word2vec_nn_f1, word2vec_nn_roc_auc, word2vec_nn_conf_matrix = nn(word2vec_X_train, word2vec_X_test, word2vec_y_train, word2vec_y_test)
# word2vec_accuracy = [word2vec_dt_accuracy,word2vec_gbc_accuracy,word2vec_knn_accuracy,word2vec_lg_accuracy,word2vec_xgb_accuracy,word2vec_nb_accuracy,word2vec_rf_accuracy,word2vec_svm_accuracy,word2vec_nn_accuracy]
# word2vec_precision = [word2vec_dt_precision,word2vec_gbc_precision,word2vec_knn_precision,word2vec_lg_precision,word2vec_xgb_precision,word2vec_nb_precision,word2vec_rf_precision,word2vec_svm_precision,word2vec_nn_precision]
# word2vec_recall = [word2vec_dt_recall,word2vec_gbc_recall,word2vec_knn_recall,word2vec_lg_recall,word2vec_xgb_recall,word2vec_nb_recall,word2vec_rf_recall,word2vec_svm_recall,word2vec_nn_recall]
# word2vec_f1 = [word2vec_dt_f1,word2vec_gbc_f1,word2vec_knn_f1,word2vec_lg_f1,word2vec_xgb_f1,word2vec_nb_f1,word2vec_rf_f1,word2vec_svm_f1,word2vec_nn_f1]
# word2vec_roc_auc = [word2vec_dt_roc_auc,word2vec_gbc_roc_auc,word2vec_knn_roc_auc,word2vec_lg_roc_auc,word2vec_xgb_roc_auc,word2vec_nb_roc_auc,word2vec_rf_roc_auc,word2vec_svm_roc_auc,word2vec_nn_roc_auc]
# word2vec_conf_matrix = [word2vec_dt_conf_matrix,word2vec_gbc_conf_matrix,word2vec_knn_conf_matrix,word2vec_lg_conf_matrix,word2vec_xgb_conf_matrix,word2vec_nb_conf_matrix,word2vec_rf_conf_matrix,word2vec_svm_conf_matrix,word2vec_nn_conf_matrix]

# word2vec_df = pd.DataFrame({
#     'Model': models,
#     'Accuracy': word2vec_accuracy,
#     'Precision': word2vec_precision, 
#     'Recall': word2vec_recall,
#     'F1': word2vec_f1,
#     'AUC_ROC': word2vec_roc_auc,
# })
# word2vec_df.set_index('Model', inplace=True)
# word2vec_df.to_excel('word2vec_metrics_full.xlsx')

# word2vec = np.array([word2vec_accuracy, word2vec_precision, word2vec_recall, word2vec_f1, word2vec_roc_auc])
# word2vec = np.round(word2vec, 2)
# fig, ax = plt.subplots(figsize=(12, 6))
# im = ax.imshow(word2vec, cmap='YlGnBu', vmin=0, vmax=1)
# ax.set_xticks(range(len(models)), labels=models,
#               rotation=45, ha="right", rotation_mode="anchor")
# ax.set_yticks(range(len(evaluations)), labels=evaluations)
# for i in range(len(evaluations)):
#     for j in range(len(models)):
#         text_color = 'white' if word2vec[i,j] > 0.5 else 'black'  
#         ax.text(j, i, word2vec[i, j],
#                 ha="center", va="center", color=text_color)
# cbar = plt.colorbar(im)
# cbar.set_label('Score')
# ax.set_title("Word2Vec Model Performance Comparison")
# fig.tight_layout()
# plt.savefig('word2vec_metrics.png', dpi=300, bbox_inches='tight')  
# plt.show()










bert_df = pd.read_excel('FINAL_Bert_Embedding.xlsx', index_col='Model')
count_df = pd.read_excel('FINAL_Count_Vectorizer.xlsx', index_col='Model')
gemini_df = pd.read_excel('FINAL_Gemini_Embedding.xlsx', index_col='Model')
tfidf_df = pd.read_excel('FINAL_tfidf_embedding.xlsx', index_col='Model')
word2vec_df = pd.read_excel('FINAL_word2vec_Embedding.xlsx', index_col='Model')
trying = pd.read_excel('word2vec_metrics_full.xlsx', index_col = 'Model')

bert_accuracy = bert_df['Accuracy']
count_accuracy = count_df['Accuracy']
gemini_accuracy = gemini_df['Accuracy']
tfidf_accuracy = tfidf_df['Accuracy']
word2vec_accuracy = word2vec_df['Accuracy']
accuracy = np.array([bert_accuracy, count_accuracy, gemini_accuracy, tfidf_accuracy, word2vec_accuracy])

all_models = sorted(list(set(bert_df.index) 
                    | set(count_df.index) 
                    | set(gemini_df.index) 
                    | set(tfidf_df.index) 
                    | set(word2vec_df.index)))

# 3. Create a consolidated DataFrame
results = pd.DataFrame(index=all_models)

# Add accuracy from each embedding method
results['BERT'] = bert_df['Accuracy']
results['Count'] = count_df['Accuracy']
results['Gemini'] = gemini_df['Accuracy']
results['TF-IDF'] = tfidf_df['Accuracy']
results['Word2Vec'] = word2vec_df['Accuracy']

# 4. Fill NaN values (for models missing in some files)
results = results.fillna(0)  # Or use another appropriate value

# 5. Convert to numpy array for plotting
accuracy = results.T.values  # Transpose to get embeddings as rows


accuracy = np.round(accuracy, 2)
fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(accuracy, cmap='YlGnBu', vmin=0.4, vmax=1)
ax.set_xticks(range(len(models)), labels=models,
              rotation=45, ha="right", rotation_mode="anchor")
ax.set_yticks(range(len(embeddings)), labels=embeddings)
for i in range(len(embeddings)):
    for j in range(len(models)):
        text_color = 'white' if accuracy[i,j] > 0.5 else 'black'  
        ax.text(j, i, accuracy[i, j],
                ha="center", va="center", color=text_color)
cbar = plt.colorbar(im)
cbar.set_label('Score')
ax.set_title("Model Accuracy Performance Comparison")
fig.tight_layout()
plt.savefig('accuracy.png', dpi=300, bbox_inches='tight')  
plt.show()

bert_precision = bert_df['Precision']
count_precision = count_df['Precision']
gemini_precision = gemini_df['Precision']
tfidf_precision = tfidf_df['Precision']
word2vec_precision = word2vec_df['Precision']

precision = np.array([bert_precision, count_precision, gemini_precision, tfidf_precision, word2vec_precision])
precision = np.round(precision, 2)
fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(precision, cmap='YlGnBu', vmin=0.5, vmax=1)
ax.set_xticks(range(len(models)), labels=models,
              rotation=45, ha="right", rotation_mode="anchor")
ax.set_yticks(range(len(embeddings)), labels=embeddings)
for i in range(len(embeddings)):
    for j in range(len(models)):
        text_color = 'white' if precision[i,j] > 0.5 else 'black'  
        ax.text(j, i, precision[i, j],
                ha="center", va="center", color=text_color)
cbar = plt.colorbar(im)
cbar.set_label('Score')
ax.set_title("Model Precision Performance Comparison")
fig.tight_layout()
plt.savefig('precision.png', dpi=300, bbox_inches='tight')  
plt.show()


bert_recall = bert_df['Recall']
count_recall = count_df['Recall']
gemini_recall = gemini_df['Recall']
tfidf_recall = tfidf_df['Recall']
word2vec_recall = word2vec_df['Recall']

recall = np.array([bert_recall, count_recall, gemini_recall, tfidf_recall, word2vec_recall])
recall = np.round(recall, 2)
fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(recall, cmap='YlGnBu', vmin=0.5, vmax=1)
ax.set_xticks(range(len(models)), labels=models,
              rotation=45, ha="right", rotation_mode="anchor")
ax.set_yticks(range(len(embeddings)), labels=embeddings)
for i in range(len(embeddings)):
    for j in range(len(models)):
        text_color = 'white' if recall[i,j] > 0.5 else 'black'  
        ax.text(j, i, recall[i, j],
                ha="center", va="center", color=text_color)
cbar = plt.colorbar(im)
cbar.set_label('Score')
ax.set_title("Model Recall Performance Comparison")
fig.tight_layout()
plt.savefig('recall.png', dpi=300, bbox_inches='tight')  
plt.show()


bert_f1 = bert_df['F1']
count_f1 = count_df['F1']
gemini_f1 = gemini_df['F1']
tfidf_f1 = tfidf_df['F1']
word2vec_f1 = word2vec_df['F1']

f1 = np.array([bert_f1, count_f1, gemini_f1, tfidf_f1, word2vec_f1])
f1 = np.round(f1, 2)
fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(f1, cmap='YlGnBu', vmin=0.5, vmax=1)
ax.set_xticks(range(len(models)), labels=models,
              rotation=45, ha="right", rotation_mode="anchor")
ax.set_yticks(range(len(embeddings)), labels=embeddings)
for i in range(len(embeddings)):
    for j in range(len(models)):
        text_color = 'white' if f1[i,j] > 0.5 else 'black'  
        ax.text(j, i, f1[i, j],
                ha="center", va="center", color=text_color)
cbar = plt.colorbar(im)
cbar.set_label('Score')
ax.set_title("Model F1 Performance Comparison")
fig.tight_layout()
plt.savefig('f1.png', dpi=300, bbox_inches='tight')  
plt.show()



bert_recall = bert_df['ROC AUC']
count_recall = count_df['ROC AUC']
gemini_recall = gemini_df['ROC AUC']
tfidf_recall = tfidf_df['ROC AUC']
word2vec_recall = word2vec_df['ROC AUC']

roc_auc = np.array([bert_recall, count_recall, gemini_recall, tfidf_recall, word2vec_recall])
roc_auc = np.round(roc_auc, 2)
fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(roc_auc, cmap='YlGnBu', vmin=0.5, vmax=1)
ax.set_xticks(range(len(models)), labels=models,
              rotation=45, ha="right", rotation_mode="anchor")
ax.set_yticks(range(len(embeddings)), labels=embeddings)
for i in range(len(embeddings)):
    for j in range(len(models)):
        text_color = 'white' if roc_auc[i,j] > 0.5 else 'black'  
        ax.text(j, i, roc_auc[i, j],
                ha="center", va="center", color=text_color)
cbar = plt.colorbar(im)
cbar.set_label('Score')
ax.set_title("Model AUC-ROC Performance Comparison")
fig.tight_layout()
plt.savefig('roc_auc.png', dpi=300, bbox_inches='tight')  
plt.show()

