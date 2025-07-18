import pandas as pd
import time
from openai import OpenAI
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

client = OpenAI(api_key="",
                base_url="https://api.deepseek.com/v1")

df_unclean = pd.read_csv('reddit_anxiety.csv')  
df = df_unclean.drop(columns=['author','id'])
# df = df.iloc[0:3]
posts = df['selftext'].tolist()

classification_prompt = """
Analyze the following text and classify it as:
- 1 if it indicates anxiety or mental health concerns
- 0 if it appears neutral or positive
Text: {post}
Only respond with either 0 or 1, no other text.
"""

results = []

for post in posts:
    formatted_prompt = classification_prompt.format(post = post)
    response = client.chat.completions.create(
    model="deepseek-chat",
    messages = [{'role':'user','content': formatted_prompt}]
)
    # time.sleep(3) #cuz i hit the rate limit for the free tier of Google's Gemini API
    classification_result = int(response.choices[0].message.content)  # Returns '0' or '1'
    results.append(classification_result)

y_predicted = results
y = df['majority_vote']

accuracy = accuracy_score(y, y_predicted)
precision = precision_score(y, y_predicted)
recall = recall_score(y, y_predicted)
f1 = f1_score(y, y_predicted)
roc_auc = roc_auc_score(y, y_predicted)
conf_matrix = confusion_matrix(y, y_predicted)

print(accuracy, precision, recall, f1, roc_auc)