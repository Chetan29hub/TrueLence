#!/usr/bin/env python3
"""
Test script for the Fake News Detection Flask Application
"""

import requests
import json

def test_homepage():
    """Test the homepage."""
    print("🧪 Testing homepage...")
    try:
        response = requests.get('http://localhost:5000')
        if response.status_code == 200:
            print("✅ Homepage accessible")
            return True
        else:
            print(f"❌ Homepage error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Homepage connection error: {e}")
        return False

def test_api():
    """Test the analysis API."""
    print("🧪 Testing API endpoint...")

    test_text = "This is a breaking news story about a major political scandal involving corruption and bribery."

    try:
        response = requests.post('http://localhost:5000/analyze',
                               json={'text': test_text, 'model': 'Logistic Regression'},
                               headers={'Content-Type': 'application/json'})

        if response.status_code == 200:
            result = response.json()
            prediction = result.get('prediction', 'Unknown')
            confidence = result.get('confidence', 0)

            print(f"✅ API working - Prediction: {prediction}")
            print(".2f")
            return True
        else:
            print(f"❌ API error: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        print(f"❌ API connection error: {e}")
        return False

def test_multiple_models():
    """Test multiple model analysis."""
    print("🧪 Testing multiple models...")

    test_text = "Scientists discover new evidence of climate change impacts."

    try:
        response = requests.post('http://localhost:5000/analyze-multiple',
                               json={'text': test_text},
                               headers={'Content-Type': 'application/json'})

        if response.status_code == 200:
            results = response.json()
            print("✅ Multiple models working:")
            for model, result in results.items():
                pred = result.get('prediction', 'Unknown')
                conf = result.get('confidence', 0)
                print(f"   {model}: {pred} ({conf:.2f})")
            return True
        else:
            print(f"❌ Multiple models error: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Multiple models connection error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Fake News Detection Flask Application")
    print("=" * 50)

    # Run tests
    homepage_ok = test_homepage()
    api_ok = test_api()
    multiple_ok = test_multiple_models()

    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"Homepage: {'✅ PASS' if homepage_ok else '❌ FAIL'}")
    print(f"API: {'✅ PASS' if api_ok else '❌ FAIL'}")
    print(f"Multiple Models: {'✅ PASS' if multiple_ok else '❌ FAIL'}")

    if all([homepage_ok, api_ok, multiple_ok]):
        print("\n🎉 All tests passed! The application is working correctly.")
        print("🌐 Access it at: http://localhost:5000")
    else:
        print("\n⚠️ Some tests failed. Check the Flask application logs.")