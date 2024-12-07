import requests
import json

# URL for the Flask API
url = "http://127.0.0.1:5000/predict"

# JSON payload with exactly 562 features
payload = {
    "features": [0.2, -0.3, 0.1, -0.5, 0.8, 0.1, -0.2, -0.3, 0.7, 0.5] + [0] * (562 - 10)
}

try:
    # Send POST request
    response = requests.post(url, json=payload)
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")
