from functions import *
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


df = pd.read_csv('reddit_anxiety.csv')
df['clean_text'] = df['selftext'].fillna('')
df['clean_text'] = df.apply(
    lambda x: x['title'] if x['clean_text'] == '' else x['clean_text'],
    axis=1
)
texts = df['clean_text']
labels = df['majority_vote']

# 2. Split into first 4000 (train) and last 1000 (test)
train_texts = texts.iloc[:2000]
train_labels = labels.iloc[:2000]
test_texts = texts.iloc[2000:4000]
test_labels = labels.iloc[2000:4000]
new_texts = texts.iloc[4000:5000]
new_labels = labels.iloc[4000:5000]
# Balance training set (2000 each)
X_train_unreadable, y_train = balance_data(train_texts, train_labels, 1000)

# Balance test set (500 each)
X_test_unreadable, y_test = balance_data(test_texts, test_labels, 1000)

# Balance test set (500 each)
X_new_unreadable, y_new = balance_data(test_texts, test_labels, 500)

genai.configure(api_key="AIzaSyBpgrGwUVhU9tTZNkRQ5RyMVBYlDRe5TAI") 
def embedding(one_post):
    model="models/embedding-001"
    return genai.embed_content(model=model, content=one_post)['embedding']

X_train = []
for post_i in tqdm(X_train_unreadable, desc="Progress"):  # Add progress bar
    X_train.append(embedding(str(post_i)))
X_test = []
for post_i in tqdm(X_test_unreadable, desc="Progress"):  # Add progress bar
    X_test.append(embedding(str(post_i)))
X_new = []
for post_i in tqdm(X_new_unreadable, desc="Progress"):  # Add progress bar
    X_new.append(embedding(str(post_i)))

#no naive base
lg = LogisticRegression(max_iter=1000, class_weight='balanced')
lg.fit(X_train, y_train)
lg_predict = lg.predict(X_test)
dt = DecisionTreeClassifier(max_depth=5)
dt.fit(X_train, y_train)
dt_predict = dt.predict(X_test)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
rf_predict = rf.predict(X_test)
# gb = GradientBoostingClassifier(n_estimators=50)
# gb.fit(X_train, y_train)
# gb_predict = gb.predict(X_test)
kn = KNeighborsClassifier(n_neighbors=5)
kn.fit(X_train, y_train)
kn_predict = kn.predict(X_test)
svc = SVC(kernel='linear', probability=True)
svc.fit(X_train, y_train)
svc_predict = svc.predict(X_test)
xg = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xg.fit(X_train, y_train)
xg_predict = xg.predict(X_test)
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500)
mlp.fit(X_train, y_train)
mlp_predict = mlp.predict(X_test)

X_val_meta = np.column_stack((lg_predict, dt_predict, rf_predict, kn_predict, xg_predict, svc_predict, mlp_predict))
meta_model = GradientBoostingClassifier(n_estimators=50)
print(np.shape(X_val_meta))
print(np.shape(y_test))
meta_model.fit(X_val_meta, y_test)


lg_predict_new = lg.predict(X_new)
dt_predict_new = dt.predict(X_new)
rf_predict_new = rf.predict(X_new)
kn_predict_new = kn.predict(X_new)
svc_predict_new = svc.predict(X_new)
xg_predict_new= xg.predict(X_new)
mlp_predict_new = mlp.predict(X_new)

X_new_meta = np.column_stack((lg_predict_new, dt_predict_new, rf_predict_new, kn_predict_new, svc_predict_new, xg_predict_new, mlp_predict_new))
y_new_predict = meta_model.predict(X_new_meta)

print(accuracy_score(y_new, y_new_predict))
#0.813