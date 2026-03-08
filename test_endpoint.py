import requests
import json

def test_predict(text):
    url = "http://127.0.0.1:5000/predict"
    payload = {"text": text}
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        print(f"Testing: '{text}'")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print("-" * 30)
    except Exception as e:
        print(f"Error testing '{text}': {e}")

if __name__ == "__main__":
    test_predict("Hello")
    test_predict("How are you?")
    test_predict("Nawaz Sharif")
    test_predict("Kutta")
