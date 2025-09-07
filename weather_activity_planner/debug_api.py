#!/usr/bin/env python3
"""
Simple test script to debug the API issues
"""
import requests
import json

def test_api():
    payload = {
        "city_query": "Amsterdam",
        "date_from": "2025-09-07T00:00:00",
        "date_to": "2025-09-09T23:59:59",
        "preferences": {}
    }
    
    try:
        print("Testing API endpoint...")
        response = requests.post(
            "http://localhost:8000/plan", 
            json=payload, 
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print("Success!")
            print(json.dumps(data, indent=2, default=str))
        else:
            print(f"Error: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error details: {json.dumps(error_data, indent=2)}")
            except Exception:
                print(f"Raw response: {response.text}")
                
    except requests.exceptions.ConnectionError:
        print("Connection error - is the API server running on port 8000?")
    except requests.exceptions.Timeout:
        print("Request timed out")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    test_api()
