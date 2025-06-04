# init_db.py - Database initialization script
import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from cryptography.fernet import Fernet
from datetime import datetime
import argparse

# Add current directory to path to import main modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from main import Base, User, FaceRegistration, FaceVerification
    print("✅ Successfully imported database models")
except ImportError as e:
    print(f"❌ Could not import models: {e}")
    print("Make sure you're running this script from the project root directory")
    sys.exit(1)

def create_database_tables(database_url):
    """Create all database tables"""
    try:
        # For SQLite, we need to handle foreign keys properly
        if database_url.startswith("sqlite"):
            engine = create_engine(database_url, echo=True, connect_args={"check_same_thread": False})
        else:
            engine = create_engine(database_url, echo=True)
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully")
        
        return engine
    except Exception as e:
        print(f"❌ Error creating database tables: {e}")
        return None

def create_test_users(engine):
    """Create test users for development"""
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    
    try:
        # Check if users already exist
        existing_users = db.query(User).count()
        if existing_users > 0:
            print(f"ℹ️  Found {existing_users} existing users, skipping test user creation")
            return
        
        test_users = [
            User(
                id=1,
                email="alice@university.edu",
                name="Alice Johnson",
                hashed_password="hashed_password_here",
                is_active=True
            ),
            User(
                id=2, 
                email="bob@university.edu",
                name="Bob Smith",
                hashed_password="hashed_password_here",
                is_active=True
            ),
            User(
                id=3,
                email="carol@university.edu", 
                name="Carol Davis",
                hashed_password="hashed_password_here",
                is_active=True
            ),
            User(
                id=4,
                email="david@university.edu",
                name="David Wilson", 
                hashed_password="hashed_password_here",
                is_active=True
            )
        ]
        
        for user in test_users:
            db.add(user)
        
        db.commit()
        print(f"✅ Created {len(test_users)} test users")
        
    except Exception as e:
        print(f"❌ Error creating test users: {e}")
        db.rollback()
    finally:
        db.close()

def check_system_requirements():
    """Check if all system requirements are met"""
    print("\n🔍 Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major == 3 and python_version.minor >= 8:
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"❌ Python version {python_version.major}.{python_version.minor} not supported. Need Python 3.8+")
        return False
    
    # Check required packages
    required_packages = [
        'face_recognition',
        'cv2',
        'PIL',
        'numpy',
        'sqlalchemy',
        'fastapi',
        'cryptography'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def generate_encryption_key():
    """Generate a new encryption key for face embeddings"""
    key = Fernet.generate_key()
    print(f"\n🔐 Generated new encryption key:")
    print(f"FACE_ENCRYPTION_KEY={key.decode()}")
    print("\n⚠️  Save this key securely! You'll need it to decrypt existing face data.")
    print("Add it to your .env file or environment variables.")
    return key.decode()

def test_database_connection(database_url):
    """Test database connection"""
    try:
        engine = create_engine(database_url)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
        print("✅ Database connection successful")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Initialize LMS Face Recognition Database")
    parser.add_argument(
        "--database-url", 
        default=os.getenv("DATABASE_URL", "sqlite:///./lms_face.db"),
        help="Database URL (default: sqlite:///./lms_face.db)"
    )
    parser.add_argument(
        "--create-test-users", 
        action="store_true",
        help="Create test users for development"
    )
    parser.add_argument(
        "--generate-key",
        action="store_true", 
        help="Generate new encryption key"
    )
    parser.add_argument(
        "--check-requirements",
        action="store_true",
        help="Check system requirements only"
    )
    
    args = parser.parse_args()
    
    print("🚀 LMS Face Recognition - Database Initialization")
    print("=" * 50)
    
    # Check requirements first
    if not check_system_requirements():
        if not args.check_requirements:
            print("\n❌ System requirements not met. Please install missing packages.")
            sys.exit(1)
        else:
            sys.exit(1)
    
    if args.check_requirements:
        print("\n✅ All system requirements met!")
        sys.exit(0)
    
    # Generate encryption key if requested
    if args.generate_key:
        key = generate_encryption_key()
        # Set the environment variable for this session
        os.environ['FACE_ENCRYPTION_KEY'] = key
    
    database_url = args.database_url
    print(f"\n📊 Database URL: {database_url}")
    
    # Test database connection
    if not test_database_connection(database_url):
        print("\n❌ Cannot connect to database. Check your DATABASE_URL.")
        sys.exit(1)
    
    # Create database tables
    print("\n🏗️  Creating database tables...")
    engine = create_database_tables(database_url)
    if not engine:
        sys.exit(1)
    
    # Create test users if requested
    if args.create_test_users:
        print("\n👥 Creating test users...")
        create_test_users(engine)
    
    # Final status
    print("\n" + "=" * 50)
    print("✅ Database initialization completed successfully!")
    print("\nNext steps:")
    print("1. Set your FACE_ENCRYPTION_KEY environment variable")
    print("2. Start the API server: uvicorn main:app --reload")
    print("3. Open the frontend HTML file in your browser")
    print("4. Test face registration and verification")
    
    if args.create_test_users:
        print("\nTest users created:")
        print("- alice@university.edu (ID: 1)")
        print("- bob@university.edu (ID: 2)") 
        print("- carol@university.edu (ID: 3)")
        print("- david@university.edu (ID: 4)")

if __name__ == "__main__":
    main()