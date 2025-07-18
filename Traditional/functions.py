import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#traditional machine learning models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

#data analysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.utils import shuffle

#embedding
import google.generativeai as genai
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from transformers import BertTokenizer, BertModel
import torch


def decision_tree(X_train, X_test, y_train, y_test):
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)
    y_predicted = tree.predict(X_test)

    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted)
    recall = recall_score(y_test, y_predicted)
    f1 = f1_score(y_test, y_predicted)
    roc_auc = roc_auc_score(y_test, y_predicted)
    conf_matrix = confusion_matrix(y_test, y_predicted)
    return accuracy, precision, recall, f1, roc_auc, conf_matrix
def gradient_boosting_classifier(X_train, X_test, y_train, y_test):
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)

    # Define Gradient Boosting Classifier with hyperparameters
    gbc=GradientBoostingClassifier(n_estimators=500,learning_rate=0.5,random_state=100,max_features=5 )
    # Fit train data to GBC
    gbc.fit(X_train_transformed, y_train)
    GradientBoostingClassifier(criterion='friedman_mse', init=None,
                            learning_rate=0.5, loss='deviance', max_depth=3,
                            max_features=5, max_leaf_nodes=None,
                            min_impurity_decrease=0.0, 
                            min_samples_leaf=1, min_samples_split=2,
                            min_weight_fraction_leaf=0.0, n_estimators=500,
                            n_iter_no_change=None, 
                            random_state=100, subsample=1.0, tol=0.0001,
                            validation_fraction=0.1, verbose=0,
                            warm_start=False)
    y_predicted = gbc.predict(X_test_transformed)

    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted)
    recall = recall_score(y_test, y_predicted)
    f1 = f1_score(y_test, y_predicted)
    roc_auc = roc_auc_score(y_test, y_predicted)
    conf_matrix = confusion_matrix(y_test, y_predicted)
    return accuracy, precision, recall, f1, roc_auc, conf_matrix
def knn(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=20)
    knn.fit(X_train, y_train)
    y_predicted = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted)
    recall = recall_score(y_test, y_predicted)
    f1 = f1_score(y_test, y_predicted)
    roc_auc = roc_auc_score(y_test, y_predicted)
    conf_matrix = confusion_matrix(y_test, y_predicted)
    return accuracy, precision, recall, f1, roc_auc, conf_matrix
def Logistic_Regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted)
    recall = recall_score(y_test, y_predicted)
    f1 = f1_score(y_test, y_predicted)
    roc_auc = roc_auc_score(y_test, y_predicted)
    conf_matrix = confusion_matrix(y_test, y_predicted)
    return accuracy, precision, recall, f1, roc_auc, conf_matrix
def XGBoost(X_train, X_test, y_train, y_test):
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
        'seed': 42}
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
    return accuracy, precision, recall, f1, roc_auc, conf_matrix
def naive_bayes(X_train, X_test, y_train, y_test):
    gnb = GaussianNB()
    y_predicted = gnb.fit(X_train, y_train).predict(X_test)

    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted)
    recall = recall_score(y_test, y_predicted)
    f1 = f1_score(y_test, y_predicted)
    roc_auc = roc_auc_score(y_test, y_predicted)
    conf_matrix = confusion_matrix(y_test, y_predicted)
    return accuracy, precision, recall, f1, roc_auc, conf_matrix
def random_forest(X_train, X_test, y_train, y_test):
    tree = RandomForestClassifier()
    tree.fit(X_train, y_train)
    y_predicted = tree.predict(X_test)

    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted)
    recall = recall_score(y_test, y_predicted)
    f1 = f1_score(y_test, y_predicted)
    roc_auc = roc_auc_score(y_test, y_predicted)
    conf_matrix = confusion_matrix(y_test, y_predicted)
    return accuracy, precision, recall, f1, roc_auc, conf_matrix
def SVM(X_train, X_test, y_train, y_test):
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted)
    recall = recall_score(y_test, y_predicted)
    f1 = f1_score(y_test, y_predicted)
    roc_auc = roc_auc_score(y_test, y_predicted)
    conf_matrix = confusion_matrix(y_test, y_predicted)
    return accuracy, precision, recall, f1, roc_auc, conf_matrix
def nn(X_train, X_test, y_train, y_test):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(4, 4), random_state=1)
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)

    accuracy = accuracy_score( y_test, y_predicted)
    precision = precision_score(y_test, y_predicted)
    recall = recall_score(y_test, y_predicted)
    f1 = f1_score(y_test, y_predicted)
    roc_auc = roc_auc_score(y_test, y_predicted)
    conf_matrix = confusion_matrix(y_test, y_predicted)
    return accuracy, precision, recall, f1, roc_auc, conf_matrix
def count_vectorizer(size):
    # df_unclean = pd.read_csv('reddit_anxiety.csv')  
    # df = df_unclean.drop(columns=['author','id'])
    # df = df.iloc[0:size]
    # df = df.dropna(subset=['selftext'])  # Remove rows where 'selftext' is NaN
    # x = df['selftext'].tolist()
    # y = df['majority_vote']

    # vectorizer = CountVectorizer()
    # count_embeddings = vectorizer.fit_transform(x)

    # np.save('count_vectorizer_embeddings.npy', count_embeddings.toarray())    
    # y.to_csv('count_vectorizer_labels.csv', index=False)
    X = np.load('count_vectorizer_embeddings.npy')            
    y = pd.read_csv('count_vectorizer_labels.csv')['majority_vote']   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=36)
    return X_train, X_test, y_train, y_test
def gemini_embedding(size):
    # df_unclean = pd.read_csv('reddit_anxiety.csv')  
    # df = df_unclean.drop(columns=['author','id'])
    # df = df.iloc[0:size]
    # x = df['selftext'].tolist()
    # y = df['majority_vote']

    # genai.configure(api_key="AIzaSyBpgrGwUVhU9tTZNkRQ5RyMVBYlDRe5TAI") 
    # def embedding(one_post):
    #     model="models/embedding-001"
    #     return genai.embed_content(model=model, content=one_post)['embedding']

    # embeddings = []
    # for post_i in tqdm(x, desc="Progress"):  # Add progress bar
    #     embeddings.append(embedding(str(post_i)))

    # np.save('gemini_embeddings.npy', embeddings)    
    # y.to_csv('gemini_labels.csv', index=False)     
    X = np.load('gemini_embeddings.npy')            
    y = pd.read_csv('gemini_labels.csv')['majority_vote']   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=36)
    return X_train, X_test, y_train, y_test 
def tfidf(size):
    # df_unclean = pd.read_csv('reddit_anxiety.csv')  
    # df = df_unclean.drop(columns=['author','id'])
    # df = df.iloc[0:size]
    # df = df.dropna(subset=['selftext'])  # Remove rows where 'selftext' is NaN
    # x = df['selftext'].tolist()
    # y = df['majority_vote']

    # vectorizer = TfidfVectorizer()
    # tfidf_embeddings = vectorizer.fit_transform(x)

    # np.save('tfidf_embeddings.npy', tfidf_embeddings.toarray())    
    # y.to_csv('tfidf_labels.csv', index=False)      
    X = np.load('tfidf_embeddings.npy')            
    y = pd.read_csv('tfidf_labels.csv')['majority_vote']   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=36)
    return X_train, X_test, y_train, y_test 
def word2vec(size):
#     df_unclean = pd.read_csv('reddit_anxiety.csv')  
#     df = df_unclean.drop(columns=['author','id'])
#     df = df.iloc[0:size]
#     df = df.dropna(subset=['selftext'])  # Remove rows where 'selftext' is NaN
#     y = df['majority_vote']    
#     sentences = df['selftext'].fillna('').tolist()

#     tokenized_sentences = [simple_preprocess(str(sent), deacc=True) for sent in sentences]

#     model = Word2Vec(
#         sentences=tokenized_sentences,
#         vector_size=100,
#         window=5,
#         min_count=2,
#         workers=4,
#         epochs=10
#     )

#     model.save("word2vec_model.model")

#     def get_doc_embedding(tokens):
#         valid_words = [word for word in tokens if word in model.wv]
#         if not valid_words:
#             return np.zeros(model.vector_size)  # Return zeros if no valid words
#         return np.mean(model.wv[valid_words], axis=0)  # Average word vectors

#     doc_embeddings = [get_doc_embedding(tokens) for tokens in tokenized_sentences]

#     np.save('word2vec_embeddings.npy', np.array(doc_embeddings))

#     df['majority_vote'].to_csv('word2vec_labels.csv', index=False)
    X = np.load('word2vec_embeddings.npy')            
    y = pd.read_csv('word2vec_labels.csv')['majority_vote']   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=36)
    return X_train, X_test, y_train, y_test 
def bert_embedding(size):
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Lowercase model
#     model = BertModel.from_pretrained('bert-base-uncased')

#     df_unclean = pd.read_csv('reddit_anxiety.csv')  
#     df = df_unclean.drop(columns=['author','id'])
#     df = df.iloc[0:size]
#     x = df['selftext'].tolist()
#     y = df['majority_vote']

#     for i in tqdm(range(int(len(x)))):
#         try:
#             inputs = tokenizer(x, return_tensors="pt", padding=True, truncation=True)
#         except ValueError as e:
#             x = [str(text) for text in x]
#             inputs = tokenizer(x, return_tensors="pt", padding=True, truncation=True)

#     with torch.no_grad():
#         outputs = model(**inputs)

#     token_embeddings = outputs.last_hidden_state
#     sentence_embeddings = torch.mean(token_embeddings, dim=1)  # Shape: [batch_size, 768]

    # np.save('bert_embeddings.npy', sentence_embeddings.numpy())
    # y.to_csv('bert_labels.csv', index=False)      
    X = np.load('bert_embeddings.npy')            
    y = pd.read_csv('bert_labels.csv')['majority_vote']   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=36)
    return X_train, X_test, y_train, y_test 



def balance_data(texts, labels, target_count):
    # Separate classes
    class_1 = texts[labels == 1]
    class_0 = texts[labels == 0]
    
    # Handle class imbalance
    if len(class_1) > target_count:
        class_1 = class_1.iloc[:target_count]  # Truncate excess class 1
    elif len(class_1) < target_count:
        needed = target_count - len(class_1)
        class_1 = pd.concat([class_1, class_1.sample(needed, replace=True)])  # Replicate
    
    if len(class_0) > target_count:
        class_0 = class_0.iloc[:target_count]  # Truncate excess class 0
    elif len(class_0) < target_count:
        needed = target_count - len(class_0)
        class_0 = pd.concat([class_0, class_0.sample(needed, replace=True)])  # Replicate
    
    # Combine and shuffle
    balanced_texts = pd.concat([class_1, class_0])
    balanced_labels = pd.Series([1]*target_count + [0]*target_count)
    
    return shuffle(balanced_texts, balanced_labels, random_state=42)


