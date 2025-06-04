# fix_db_sync.py - Fix database synchronization issue
import os
import sqlite3
from pathlib import Path

def check_database_files():
    """Find all database files in current directory and subdirectories"""
    print("🔍 Searching for database files...")
    
    db_files = []
    
    # Check current directory
    for pattern in ["*.db", "*.sqlite", "*.sqlite3"]:
        files = list(Path(".").glob(pattern))
        db_files.extend(files)
    
    # Check subdirectories
    for pattern in ["**/*.db", "**/*.sqlite", "**/*.sqlite3"]:
        files = list(Path(".").glob(pattern))
        db_files.extend(files)
    
    if db_files:
        print(f"Found {len(db_files)} database files:")
        for db_file in db_files:
            print(f"  📁 {db_file.absolute()}")
            try:
                # Check if it has our tables
                conn = sqlite3.connect(str(db_file))
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [table[0] for table in cursor.fetchall()]
                conn.close()
                print(f"     Tables: {tables}")
            except Exception as e:
                print(f"     Error reading: {e}")
    else:
        print("❌ No database files found!")
    
    return db_files

def fix_database_path():
    """Ensure database is in the correct location with correct content"""
    print("\n🔧 Fixing database path and content...")
    
    # Target database file
    target_db = "lms_face.db"
    
    # Remove any existing database
    if Path(target_db).exists():
        Path(target_db).unlink()
        print(f"✅ Removed existing {target_db}")
    
    # Create fresh database with tables
    conn = sqlite3.connect(target_db)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute("""
        CREATE TABLE users (
            id INTEGER NOT NULL PRIMARY KEY,
            email VARCHAR UNIQUE,
            name VARCHAR,
            hashed_password VARCHAR,
            created_at DATETIME,
            is_active BOOLEAN
        )
    """)
    
    # Create face_registrations table
    cursor.execute("""
        CREATE TABLE face_registrations (
            id VARCHAR(36) NOT NULL PRIMARY KEY,
            user_id INTEGER,
            face_embedding BLOB,
            registration_photo_path VARCHAR,
            quality_score FLOAT,
            face_confidence FLOAT,
            created_at DATETIME,
            is_active BOOLEAN
        )
    """)
    
    # Create face_verifications table
    cursor.execute("""
        CREATE TABLE face_verifications (
            id VARCHAR(36) NOT NULL PRIMARY KEY,
            user_id INTEGER,
            quiz_id VARCHAR,
            verification_photo_path VARCHAR,
            similarity_score FLOAT,
            verification_result BOOLEAN,
            threshold_used FLOAT,
            created_at DATETIME,
            ip_address VARCHAR,
            user_agent TEXT
        )
    """)
    
    # Create indexes
    cursor.execute("CREATE UNIQUE INDEX ix_users_email ON users (email)")
    cursor.execute("CREATE INDEX ix_users_id ON users (id)")
    cursor.execute("CREATE INDEX ix_face_registrations_user_id ON face_registrations (user_id)")
    cursor.execute("CREATE INDEX ix_face_verifications_user_id ON face_verifications (user_id)")
    
    # Insert test users
    test_users = [
        (1, 'abhishek@university.edu', 'Abhishek Kumar', 'hashed_password_here', '2025-05-30 17:15:36', 1),
        (2, 'babita@university.edu', 'Babita Singh', 'hashed_password_here', '2025-05-30 17:15:36', 1),
        (3, 'chaman@university.edu', 'Chaman Singh', 'hashed_password_here', '2025-05-30 17:15:36', 1),
        (4, 'dhawan@university.edu', 'Dhawan Raj', 'hashed_password_here', '2025-05-30 17:15:36', 1),
    ]
    
    cursor.executemany(
        "INSERT INTO users (id, email, name, hashed_password, created_at, is_active) VALUES (?, ?, ?, ?, ?, ?)",
        test_users
    )
    
    conn.commit()
    conn.close()
    
    print(f"✅ Created fresh {target_db} with all tables and test users")
    
    # Verify tables exist
    conn = sqlite3.connect(target_db)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [table[0] for table in cursor.fetchall()]
    conn.close()
    
    print(f"✅ Verified tables exist: {tables}")
    
    return target_db

def test_database_access():
    """Test database access using the same method as the server"""
    print("\n🧪 Testing database access...")
    
    try:
        from sqlalchemy import create_engine, text
        from sqlalchemy.orm import sessionmaker
        
        # Use exact same connection method as server
        database_url = "sqlite:///./lms_face.db"
        engine = create_engine(database_url, echo=False, connect_args={"check_same_thread": False})
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';"))
            tables = [row[0] for row in result]
            print(f"✅ SQLAlchemy can see tables: {tables}")
        
        # Test ORM session
        db = SessionLocal()
        result = db.execute(text("SELECT COUNT(*) FROM users"))
        user_count = result.fetchone()[0]
        print(f"✅ Found {user_count} users via ORM")
        
        result = db.execute(text("SELECT COUNT(*) FROM face_registrations"))
        reg_count = result.fetchone()[0]
        print(f"✅ Found {reg_count} face_registrations via ORM")
        
        db.close()
        
        print("✅ Database access test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Database access test failed: {e}")
        return False

def create_env_file():
    """Create .env file with correct settings"""
    print("\n📝 Creating .env file...")
    
    env_content = f"""FACE_ENCRYPTION_KEY=LLIx2GGtnOS_gXVQIkjAvfZzQqK-vHKYcKraxPLwqkc=
DATABASE_URL=sqlite:///./lms_face.db
AUTH_TOKEN=lms-face-token-123
SIMILARITY_THRESHOLD=0.6
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ Created .env file with correct settings")

def main():
    print("🔧 Database Synchronization Fix")
    print("=" * 40)
    
    # Check current state
    check_database_files()
    
    # Fix database
    target_db = fix_database_path()
    
    # Test access
    success = test_database_access()
    
    # Create env file
    create_env_file()
    
    print("\n" + "=" * 40)
    if success:
        print("✅ Database synchronization fix completed!")
        print("\nNext steps:")
        print("1. Set environment variables:")
        print("   export FACE_ENCRYPTION_KEY=LLIx2GGtnOS_gXVQIkjAvfZzQqK-vHKYcKraxPLwqkc=")
        print("   export DATABASE_URL=sqlite:///./lms_face.db")
        print("2. Start server:")
        print("   uvicorn main:app --reload")
        print("3. Test registration in frontend")
    else:
        print("❌ Database synchronization fix failed")

if __name__ == "__main__":
    main()