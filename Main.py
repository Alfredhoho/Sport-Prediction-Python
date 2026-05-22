import requests

url = "https://api.openai.com/v1/chat/completions"
headers = {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json",
}
payload = {
    "model": "gpt-4o-mini",
    "messages": [
        {
            "role": "system",
            "content": "You are a football analyst that predicts European football match scores."
        },
        {
            "role": "user",
            "content": (
                "Predict the final score for the next match:\n"
                "Competition: UEFA Champions League\n"
                "Home team: Manchester City\n"
                "Away team: Bayern Munich\n"
                "Match date: 2026-05-26\n"
                "Use a realistic scoreline and explain the main reasons."
            )
        }
    ],
    "temperature": 0.7,
    "max_tokens": 200
}

response = requests.post(url, headers=headers, json=payload)
response.raise_for_status()
data = response.json()
print(data["choices"][0]["message"]["content"])
