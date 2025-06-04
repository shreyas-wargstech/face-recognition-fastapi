# force_create_db.py - Force database creation
import os
import sys
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, LargeBinary, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from cryptography.fernet import Fernet

# Create base
Base = declarative_base()

# Define models exactly as in main.py
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    name = Column(String)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class FaceRegistration(Base):
    __tablename__ = "face_registrations"
    
    id = Column(String(36), primary_key=True)
    user_id = Column(Integer, index=True)
    face_embedding = Column(LargeBinary)
    registration_photo_path = Column(String)
    quality_score = Column(Float)
    face_confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class FaceVerification(Base):
    __tablename__ = "face_verifications"
    
    id = Column(String(36), primary_key=True)
    user_id = Column(Integer, index=True)
    quiz_id = Column(String, nullable=True)
    verification_photo_path = Column(String)
    similarity_score = Column(Float)
    verification_result = Column(Boolean)
    threshold_used = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    ip_address = Column(String)
    user_agent = Column(Text)

def force_create_database():
    """Force create database with proper error handling"""
    print("🔨 Force Creating Database Tables...")
    
    # Database URL
    database_url = "sqlite:///./lms_face.db"
    print(f"Database URL: {database_url}")
    
    # Create engine
    try:
        engine = create_engine(database_url, echo=True, connect_args={"check_same_thread": False})
        print("✅ Engine created successfully")
    except Exception as e:
        print(f"❌ Failed to create engine: {e}")
        return False
    
    # Drop all tables first (clean slate)
    try:
        Base.metadata.drop_all(bind=engine)
        print("✅ Dropped existing tables (if any)")
    except Exception as e:
        print(f"⚠️  Warning dropping tables: {e}")
    
    # Create all tables
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ All tables created successfully")
    except Exception as e:
        print(f"❌ Failed to create tables: {e}")
        return False
    
    # Verify tables exist
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';"))
            tables = [row[0] for row in result]
            print(f"✅ Tables verified: {tables}")
            
            if 'face_registrations' not in tables:
                print("❌ face_registrations table missing!")
                return False
    except Exception as e:
        print(f"❌ Failed to verify tables: {e}")
        return False
    
    # Create test users
    try:
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        db = SessionLocal()
        
        # Check if users already exist
        existing_users = db.query(User).count()
        if existing_users > 0:
            print(f"ℹ️  Found {existing_users} existing users")
        else:
            test_users = [
                User(id=1, email="alice@university.edu", name="Alice Johnson", 
                     hashed_password="hashed_password_here", is_active=True),
                User(id=2, email="bob@university.edu", name="Bob Smith", 
                     hashed_password="hashed_password_here", is_active=True),
                User(id=3, email="carol@university.edu", name="Carol Davis", 
                     hashed_password="hashed_password_here", is_active=True),
                User(id=4, email="david@university.edu", name="David Wilson", 
                     hashed_password="hashed_password_here", is_active=True)
            ]
            
            for user in test_users:
                db.add(user)
            
            db.commit()
            print(f"✅ Created {len(test_users)} test users")
        
        db.close()
    except Exception as e:
        print(f"❌ Failed to create test users: {e}")
        return False
    
    # Generate encryption key
    key = Fernet.generate_key()
    print(f"\n🔐 Generated encryption key: {key.decode()}")
    print("\n⚠️  IMPORTANT: Set this environment variable:")
    print(f"export FACE_ENCRYPTION_KEY={key.decode()}")
    
    # Save to .env file
    try:
        with open('.env', 'w') as f:
            f.write(f"FACE_ENCRYPTION_KEY={key.decode()}\n")
            f.write("DATABASE_URL=sqlite:///./lms_face.db\n")
            f.write("AUTH_TOKEN=lms-face-token-123\n")
        print("✅ Created .env file with settings")
    except Exception as e:
        print(f"⚠️  Could not create .env file: {e}")
    
    return True

def test_database_operations():
    """Test basic database operations"""
    print("\n🧪 Testing Database Operations...")
    
    try:
        from sqlalchemy import create_engine, text
        from sqlalchemy.orm import sessionmaker
        
        engine = create_engine("sqlite:///./lms_face.db", connect_args={"check_same_thread": False})
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM users"))
            user_count = result.fetchone()[0]
            print(f"✅ Found {user_count} users in database")
        
        # Test session operations
        db = SessionLocal()
        users = db.query(User).all()
        print(f"✅ Successfully queried {len(users)} users via ORM")
        
        # Test face_registrations table
        registrations = db.query(FaceRegistration).all()
        print(f"✅ Successfully queried face_registrations table ({len(registrations)} records)")
        
        db.close()
        
        print("✅ All database operations successful!")
        return True
        
    except Exception as e:
        print(f"❌ Database operation failed: {e}")
        return False

if __name__ == "__main__":
    print("🔨 LMS Face Recognition - Force Database Creation")
    print("=" * 55)
    
    success = force_create_database()
    
    if success:
        test_success = test_database_operations()
        if test_success:
            print("\n🎉 Database setup completed successfully!")
            print("\nNext steps:")
            print("1. Set the environment variable:")
            print("   export FACE_ENCRYPTION_KEY=$(grep FACE_ENCRYPTION_KEY .env | cut -d'=' -f2)")
            print("2. Start the server:")
            print("   uvicorn main:app --reload")
            print("3. Test with the frontend")
        else:
            print("\n❌ Database tests failed")
    else:
        print("\n❌ Database creation failed")