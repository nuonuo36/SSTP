# import pandas as pd
# import requests
# import time
# from sklearn.metrics import accuracy_score

# # API Configuration
# API_URL = "http://127.0.0.1:8081/v1/chat/completions"
# API_KEY = "sstp2"
# HEADERS = {
#     "Content-Type": "application/json",
#     "Authorization": f"Bearer {API_KEY}"
# }

# # Load data
# df = pd.read_csv('reddit_anxiety.csv').drop(columns=['author','id']).iloc[0:10]
# posts = [str(p)[:128] for p in df['selftext'].tolist()]  # Truncate long posts
# y_true = df['majority_vote']

# results = []
# for post in posts:
#     try:
#         # Format message according to the chat template
#         messages = [
#             {
#                 "role": "system",
#                 "content": "Analyze the text and classify as 1 (anxiety) or 0 (neutral). Respond ONLY with 0 or 1."
#             },
#             {
#                 "role": "user",
#                 "content": post
#             }
#         ]
        
#         payload = {
#             "model": "Llama-2-7B-Chat",
#             "messages": messages,
#             "max_tokens": 1,
#             "temperature": 0,
#             "stop": ["\n"]
#         }
        
#         response = requests.post(API_URL, json=payload, headers=HEADERS)
        
#         if not response.ok:
#             print(f"Error {response.status_code}: {response.text}")
#             results.append(0)
#             continue
            
#         # Parse response
#         try:
#             response_text = response.json()["choices"][0]["message"]["content"].strip()
#             classification = 1 if "1" in response_text else 0
#         except:
#             classification = 0
            
#         results.append(classification)
#         time.sleep(0.5)
        
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         results.append(0)

# # Calculate metrics
# if len(results) == len(y_true):
#     print(f"Accuracy: {accuracy_score(y_true, results):.2f}")
#     print("Predictions:", results)
# else:
#     print("Prediction length mismatch")

# import pandas as pd
# import requests
# import time
# from sklearn.metrics import accuracy_score

# # API Configuration
# API_URL = "http://127.0.0.1:8080/v1/chat/completions"
# API_KEY = ""
# HEADERS = {
#     "Content-Type": "application/json"
# }

# # Load data
# df = pd.read_csv('reddit_anxiety.csv').drop(columns=['author','id']).iloc[0:100]
# posts = [str(p)[:128] for p in df['selftext'].tolist()]
# y_true = df['majority_vote']

# results = []
# for post in posts:
#     try:
#         messages = [
#             {
#                 "role": "system",
#                 "content": "Analyze the text and classify as 1 (anxiety) or 0 (neutral). Respond ONLY with 0 or 1."
#             },
#             {
#                 "role": "user",
#                 "content": post
#             }
#         ]
#         payload = {
#             "model": "Meta-Llama-3-8B-Instruct",
#             "messages": messages,
#             "max_tokens": 1,
#             "temperature": 0,
#             "stop": ["\n"]
#         }
#         response = requests.post(API_URL, json=payload, headers=HEADERS)
#         if not response.ok:
#             print(f"Error {response.status_code}: {response.text}")
#             results.append(0)
#             continue
#         try:
#             response_text = response.json()["choices"][0]["message"]["content"].strip()
#             classification = 1 if "1" in response_text else 0
#         except:
#             classification = 0
#         results.append(classification)
#         time.sleep(0.1)
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         results.append(0)

# if len(results) == len(y_true):
#     print(f"Accuracy: {accuracy_score(y_true, results):.2f}")
#     print("Predictions:", results)
# else:
#     print("Prediction length mismatch")


import pandas as pd
import requests
import time
from sklearn.metrics import accuracy_score

# API Configuration
API_URL = "http://127.0.0.1:8080/v1/chat/completions"
API_KEY = ""
HEADERS = {
    "Content-Type": "application/json"
}

# Load data
df = pd.read_csv('reddit_anxiety.csv').drop(columns=['author','id']).iloc[0:100]
posts = [str(p)[:128] for p in df['selftext'].tolist()]
y_true = df['majority_vote']

results = []
for post in posts:
    try:
        messages = [
            {
                "role": "system",
                "content": "Analyze the text and classify as 1 (anxiety) or 0 (neutral). Respond ONLY with 0 or 1."
            },
            {
                "role": "user",
                "content": post
            }
        ]
        payload = {
            "model": "Meta-Llama-3-8B-Instruct",
            "messages": messages,
            "max_tokens": 1,
            "temperature": 0,
            "stop": ["\n"]
        }
        response = requests.post(API_URL, json=payload, headers=HEADERS)
        if not response.ok:
            print(f"Error {response.status_code}: {response.text}")
            results.append(0)
            continue
        try:
            response_text = response.json()["choices"][0]["message"]["content"].strip()
            classification = 1 if "1" in response_text else 0
        except:
            classification = 0
        results.append(classification)
        time.sleep(0.1)
    except Exception as e:
        print(f"Error: {str(e)}")
        results.append(0)

if len(results) == len(y_true):
    print(f"Accuracy: {accuracy_score(y_true, results):.2f}")
    print("Predictions:", results)
else:
    print("Prediction length mismatch")
