import requests
import json

url = "http://localhost:5000/predict"
data = {"text": "tu pagal hai bilkul"}

try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
