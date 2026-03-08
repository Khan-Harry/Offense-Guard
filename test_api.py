import requests
import json

# The URL of your local API
url = "http://localhost:5000/predict"

# Sample text to test (Urdu/Roman Urdu)
data = {
    "text": "tu pagal hai bilkul"
}

print(f"Testing API at: {url}")
print(f"Sending text: {data['text']}")

try:
    # Send a POST request to the API
    response = requests.post(url, json=data)
    
    # Check if the request was successful
    if response.status_code == 200:
        print("\n✅ Success! Response:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"\n❌ Error: Status Code {response.status_code}")
        print(response.text)
except requests.exceptions.ConnectionError:
    print("\n❌ Error: Could not connect to the API.")
    print("Make sure 'python app.py' is running and MongoDB is started.")
except Exception as e:
    print(f"\n❌ An unexpected error occurred: {e}")
