# run.py - Alternative startup script in Python
import subprocess
import sys
import os
import time
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def main():
    print("🚀 LMS Face Recognition - Python Setup")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("main.py").exists():
        print("❌ main.py not found. Please run this script from the project directory.")
        sys.exit(1)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print(f"❌ Python 3.8+ required. Found: {sys.version}")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        sys.exit(1)
    
    # Initialize database
    if not run_command("python init_db.py --create-test-users", "Initializing database"):
        sys.exit(1)
    
    # Generate encryption key if .env doesn't exist
    if not Path(".env").exists():
        print("🔐 Creating .env file...")
        from cryptography.fernet import Fernet
        key = Fernet.generate_key().decode()
        
        with open(".env.example", "r") as f:
            env_content = f.read()
        
        env_content = env_content.replace("your-face-encryption-key-32-bytes", key)
        
        with open(".env", "w") as f:
            f.write(env_content)
        
        print("✅ .env file created")
    
    # Create uploads directory
    Path("uploads").mkdir(exist_ok=True)
    print("✅ Uploads directory ready")
    
    print("\n🎯 Setup completed! Starting the server...")
    print("   API: http://localhost:8000")
    print("   Docs: http://localhost:8000/docs")
    print("   Health: http://localhost:8000/api/v1/health")
    print("\n   Press Ctrl+C to stop the server")
    
    # Start the server
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n👋 Server stopped")

if __name__ == "__main__":
    main()

---