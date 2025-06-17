-- MySQL Database Setup Script
-- Create the base tables required before running the DeepFace migration
-- Run this script BEFORE running your migration script

-- ===========================================
-- Create Database (if needed)
-- ===========================================
-- CREATE DATABASE IF NOT EXISTS your_database_name;
-- USE your_database_name;

-- ===========================================
-- STEP 1: Create AppUser table
-- ===========================================
CREATE TABLE AppUser (
    id INT PRIMARY KEY AUTO_INCREMENT,
    Name VARCHAR(255) NOT NULL,
    Email VARCHAR(255) UNIQUE NOT NULL,
    Status VARCHAR(50) DEFAULT 'active',
    Active BOOLEAN DEFAULT TRUE,
    CreationDateTime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UpdateDateTime TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Additional common user fields
    Password VARCHAR(255),
    PhoneNumber VARCHAR(20),
    Role VARCHAR(50) DEFAULT 'student',
    Department VARCHAR(100),
    StudentID VARCHAR(50),
    
    -- Indexes
    INDEX idx_user_email (Email),
    INDEX idx_user_active (Active),
    INDEX idx_user_status (Status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ===========================================
-- STEP 2: Create original Face table (before migration)
-- ===========================================
CREATE TABLE Face (
    id INT PRIMARY KEY AUTO_INCREMENT,
    UserID INT NOT NULL,
    CreationDateTime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UpdateDateTime TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Basic face data (before DeepFace integration)
    FaceData LONGBLOB COMMENT 'Original face data',
    FileName VARCHAR(255),
    FileSize INT,
    
    -- Indexes
    INDEX idx_face_userid (UserID),
    
    -- Foreign key constraint
    FOREIGN KEY (UserID) REFERENCES AppUser(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ===========================================
-- STEP 3: Insert sample data for testing
-- ===========================================

-- Insert sample users
INSERT INTO AppUser (Name, Email, Status, Active, Role, Department) VALUES
('John Doe', 'john.doe@example.com', 'active', TRUE, 'student', 'Computer Science'),
('Jane Smith', 'jane.smith@example.com', 'active', TRUE, 'student', 'Engineering'),
('Bob Johnson', 'bob.johnson@example.com', 'active', TRUE, 'instructor', 'Mathematics'),
('Alice Brown', 'alice.brown@example.com', 'inactive', FALSE, 'student', 'Physics'),
('Charlie Wilson', 'charlie.wilson@example.com', 'active', TRUE, 'student', 'Computer Science');

-- Insert sample face records
INSERT INTO Face (UserID, FaceData, FileName, FileSize) VALUES
(1, 'sample_face_data_1', 'john_face.jpg', 1024),
(2, 'sample_face_data_2', 'jane_face.jpg', 2048),
(3, 'sample_face_data_3', 'bob_face.jpg', 1536);

-- ===========================================
-- STEP 4: Verify table creation
-- ===========================================

-- Check AppUser table
SELECT 'AppUser table created' as status, COUNT(*) as record_count FROM AppUser;

-- Check Face table  
SELECT 'Face table created' as status, COUNT(*) as record_count FROM Face;

-- Show table structures
DESCRIBE AppUser;
DESCRIBE Face;

-- ===========================================
-- STEP 5: Grant permissions (adjust as needed)
-- ===========================================

-- Example permission grants (adjust username and host as needed)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON your_database_name.* TO 'your_app_user'@'localhost';
-- GRANT CREATE, ALTER, INDEX, DROP ON your_database_name.* TO 'your_app_user'@'localhost';

-- ===========================================
-- DATABASE SETUP COMPLETE
-- ===========================================

-- IMPORTANT NOTES:
-- 1. After running this script, you can run your DeepFace migration script
-- 2. Adjust the database name and user credentials as needed for your environment
-- 3. The sample data is for testing purposes - remove or modify as needed
-- 4. Make sure to backup your data before running migrations in production
-- 5. Test the migration on a development database first

-- Next steps:
-- 1. Run this script to create the base tables
-- 2. Run your DeepFace migration script
-- 3. Test the integration with your application
-- 4. Set up proper backup and monitoring procedures