import pandas as pd
import time
from google import genai
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

client = genai.Client(api_key="")

df_unclean = pd.read_csv('reddit_anxiety.csv')  
df = df_unclean.drop(columns=['author','id'])
df = df.iloc[0:2000]
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

    response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=formatted_prompt
    )
    time.sleep(3) #cuz i hit the rate limit for the free tier of Google's Gemini API
    result = int(response.text.strip())
    results.append(result)

y_predicted = results
y = df['majority_vote']
accuracy = accuracy_score(y, y_predicted)
precision = precision_score(y, y_predicted)
recall = recall_score(y, y_predicted)
f1 = f1_score(y, y_predicted)
roc_auc = roc_auc_score(y, y_predicted)
conf_matrix = confusion_matrix(y, y_predicted)
print(accuracy)