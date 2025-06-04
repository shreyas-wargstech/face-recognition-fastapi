# test_setup.py - Test script to verify everything works
import requests
import json
import os
from pathlib import Path

def test_api_health():
    """Test if API is running and healthy"""
    try:
        response = requests.get("http://localhost:8000/api/v1/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is healthy")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Is the server running?")
        return False
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def test_database():
    """Test database connectivity through API"""
    try:
        response = requests.get(
            "http://localhost:8000/api/v1/face/registration/1",
            headers={"Authorization": "Bearer lms-face-token-123"},
            timeout=5
        )
        if response.status_code in [200, 404]:  # Both are OK - means DB is working
            print("✅ Database connectivity OK")
            return True
        else:
            print(f"❌ Database test failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def main():
    print("🧪 LMS Face Recognition - System Test")
    print("=" * 40)
    
    # Check if server is running
    if not test_api_health():
        print("\n💡 To start the server:")
        print("   uvicorn main:app --reload")
        return
    
    # Test database
    test_database()
    
    # Check file structure
    required_files = ["main.py", "init_db.py", "requirements.txt"]
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file} found")
        else:
            print(f"❌ {file} missing")
    
    # Check uploads directory
    if Path("uploads").exists():
        print("✅ uploads directory ready")
    else:
        print("❌ uploads directory missing")
        Path("uploads").mkdir(exist_ok=True)
        print("✅ uploads directory created")
    
    print("\n🎯 System test completed!")
    print("\nNext steps:")
    print("1. Open frontend.html in your browser")
    print("2. Try registering a face for user ID 1")
    print("3. Test verification with the same user")

if __name__ == "__main__":
    main()