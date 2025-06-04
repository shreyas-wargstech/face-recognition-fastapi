-- Insert 20 Sample Users into AppUser Table
-- This script ensures referenced tables exist and inserts diverse sample data

-- ===========================================
-- STEP 1: Ensure referenced tables exist with sample data
-- ===========================================

-- Create and populate Role table if it doesn't exist
CREATE TABLE IF NOT EXISTS Role (
    id INT PRIMARY KEY AUTO_INCREMENT,
    Name VARCHAR(100) NOT NULL UNIQUE,
    Description TEXT,
    Active BOOLEAN DEFAULT TRUE,
    CreationDateTime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UpdationDateTime TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Insert roles if they don't exist
INSERT IGNORE INTO Role (id, Name, Description, Active) VALUES
(1, 'Student', 'Student role with basic learning permissions', TRUE),
(2, 'Instructor', 'Instructor role with teaching permissions', TRUE),
(3, 'Admin', 'Administrator role with full system permissions', TRUE),
(4, 'Teaching Assistant', 'TA role with limited teaching permissions', TRUE),
(5, 'Guest', 'Guest role with read-only permissions', TRUE);

-- Create and populate Avatar table if it doesn't exist
CREATE TABLE IF NOT EXISTS Avatar (
    id INT PRIMARY KEY AUTO_INCREMENT,
    Name VARCHAR(100) NOT NULL,
    ImageUrl VARCHAR(255),
    Active BOOLEAN DEFAULT TRUE,
    CreationDateTime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UpdationDateTime TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Insert avatars if they don't exist
INSERT IGNORE INTO Avatar (id, Name, ImageUrl, Active) VALUES
(1, 'Default Male', '/images/avatars/male_default.png', TRUE),
(2, 'Default Female', '/images/avatars/female_default.png', TRUE),
(3, 'Professional', '/images/avatars/professional.png', TRUE),
(4, 'Casual Male', '/images/avatars/casual_male.png', TRUE),
(5, 'Casual Female', '/images/avatars/casual_female.png', TRUE),
(6, 'Academic', '/images/avatars/academic.png', TRUE);

-- Create and populate Specialization table if it doesn't exist
CREATE TABLE IF NOT EXISTS Specialization (
    id INT PRIMARY KEY AUTO_INCREMENT,
    Name VARCHAR(100) NOT NULL,
    Description TEXT,
    Code VARCHAR(20),
    Active BOOLEAN DEFAULT TRUE,
    CreationDateTime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UpdationDateTime TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Insert specializations if they don't exist
INSERT IGNORE INTO Specialization (id, Name, Description, Code, Active) VALUES
(1, 'Computer Science', 'Computer Science and Software Engineering', 'CS', TRUE),
(2, 'Mathematics', 'Mathematics and Applied Mathematics', 'MATH', TRUE),
(3, 'Physics', 'Physics and Applied Physics', 'PHY', TRUE),
(4, 'Engineering', 'General Engineering', 'ENG', TRUE),
(5, 'Business Administration', 'Business and Management', 'BIZ', TRUE),
(6, 'Biology', 'Biology and Life Sciences', 'BIO', TRUE),
(7, 'Chemistry', 'Chemistry and Chemical Engineering', 'CHEM', TRUE),
(8, 'Psychology', 'Psychology and Behavioral Sciences', 'PSY', TRUE),
(9, 'Data Science', 'Data Science and Analytics', 'DS', TRUE),
(10, 'Artificial Intelligence', 'AI and Machine Learning', 'AI', TRUE);

-- ===========================================
-- STEP 2: Insert 20 diverse sample users
-- ===========================================

INSERT INTO AppUser (
    Salutation, 
    Name, 
    Email, 
    MobileNumber, 
    PasswordHash, 
    RoleID, 
    AvatarID, 
    ProfilePictureUrl, 
    SpecializationID, 
    Status, 
    Active, 
    LastLoginDateTime, 
    isPopupFlag
) VALUES
-- Students (IDs 1-12)
('Mr.', 'John Smith', 'john.smith@university.edu', '+1-555-0101', '$2b$10$hashedpassword1', 1, 1, '/uploads/profiles/john_smith.jpg', 1, 'REGISTERED', TRUE, '2024-06-03 14:30:00', FALSE),

('Ms.', 'Emily Johnson', 'emily.johnson@university.edu', '+1-555-0102', '$2b$10$hashedpassword2', 1, 2, '/uploads/profiles/emily_johnson.jpg', 2, 'REGISTERED', TRUE, '2024-06-03 16:45:00', TRUE),

('Mr.', 'Michael Davis', 'michael.davis@university.edu', '+1-555-0103', '$2b$10$hashedpassword3', 1, 4, NULL, 1, 'DISPLAYIMAGEUPLOADED', TRUE, '2024-06-02 09:15:00', FALSE),

('Ms.', 'Sarah Wilson', 'sarah.wilson@university.edu', '+1-555-0104', '$2b$10$hashedpassword4', 1, 2, '/uploads/profiles/sarah_wilson.jpg', 3, 'REGISTERED', TRUE, '2024-06-03 11:20:00', FALSE),

('Mr.', 'David Brown', 'david.brown@university.edu', '+1-555-0105', '$2b$10$hashedpassword5', 1, 1, NULL, 4, 'PROFILEVERIFIED', TRUE, '2024-06-01 13:45:00', TRUE),

('Ms.', 'Jessica Miller', 'jessica.miller@university.edu', '+1-555-0106', '$2b$10$hashedpassword6', 1, 5, '/uploads/profiles/jessica_miller.jpg', 5, 'REGISTERED', TRUE, '2024-06-03 17:30:00', FALSE),

('Mr.', 'Christopher Garcia', 'christopher.garcia@university.edu', '+1-555-0107', '$2b$10$hashedpassword7', 1, 1, NULL, 6, 'DISPLAYIMAGEUPLOADED', TRUE, '2024-06-02 08:30:00', FALSE),

('Ms.', 'Amanda Rodriguez', 'amanda.rodriguez@university.edu', '+1-555-0108', '$2b$10$hashedpassword8', 1, 2, '/uploads/profiles/amanda_rodriguez.jpg', 7, 'REGISTERED', TRUE, '2024-06-03 12:15:00', TRUE),

('Mr.', 'Matthew Martinez', 'matthew.martinez@university.edu', '+1-555-0109', '$2b$10$hashedpassword9', 1, 4, NULL, 8, 'PROFILEVERIFIED', TRUE, '2024-05-30 15:45:00', FALSE),

('Ms.', 'Ashley Anderson', 'ashley.anderson@university.edu', '+1-555-0110', '$2b$10$hashedpassword10', 1, 5, '/uploads/profiles/ashley_anderson.jpg', 9, 'REGISTERED', TRUE, '2024-06-03 10:30:00', FALSE),

('Mr.', 'Joshua Taylor', 'joshua.taylor@university.edu', '+1-555-0111', '$2b$10$hashedpassword11', 1, 1, NULL, 10, 'SEEDED', FALSE, NULL, TRUE),

('Ms.', 'Stephanie Thomas', 'stephanie.thomas@university.edu', '+1-555-0112', '$2b$10$hashedpassword12', 1, 2, '/uploads/profiles/stephanie_thomas.jpg', 1, 'REGISTERED', TRUE, '2024-06-03 14:00:00', FALSE),

-- Instructors (IDs 13-17)
('Dr.', 'Robert Johnson', 'robert.johnson@university.edu', '+1-555-0201', '$2b$10$hashedpassword13', 2, 3, '/uploads/profiles/robert_johnson.jpg', 1, 'REGISTERED', TRUE, '2024-06-03 08:00:00', FALSE),

('Dr.', 'Linda Williams', 'linda.williams@university.edu', '+1-555-0202', '$2b$10$hashedpassword14', 2, 6, '/uploads/profiles/linda_williams.jpg', 2, 'REGISTERED', TRUE, '2024-06-03 07:30:00', TRUE),

('Dr.', 'James Wilson', 'james.wilson@university.edu', '+1-555-0203', '$2b$10$hashedpassword15', 2, 3, '/uploads/profiles/james_wilson.jpg', 3, 'REGISTERED', TRUE, '2024-06-02 16:45:00', FALSE),

('Dr.', 'Mary Davis', 'mary.davis@university.edu', '+1-555-0204', '$2b$10$hashedpassword16', 2, 6, '/uploads/profiles/mary_davis.jpg', 4, 'REGISTERED', TRUE, '2024-06-03 09:15:00', FALSE),

('Dr.', 'William Garcia', 'william.garcia@university.edu', '+1-555-0205', '$2b$10$hashedpassword17', 2, 3, NULL, 5, 'PROFILEVERIFIED', TRUE, '2024-05-28 14:30:00', TRUE),

-- Teaching Assistants (IDs 18-19)
('Ms.', 'Jennifer Lee', 'jennifer.lee@university.edu', '+1-555-0301', '$2b$10$hashedpassword18', 4, 2, '/uploads/profiles/jennifer_lee.jpg', 1, 'REGISTERED', TRUE, '2024-06-03 13:45:00', FALSE),

('Mr.', 'Kevin White', 'kevin.white@university.edu', '+1-555-0302', '$2b$10$hashedpassword19', 4, 1, '/uploads/profiles/kevin_white.jpg', 2, 'REGISTERED', TRUE, '2024-06-03 11:00:00', FALSE),

-- Admin (ID 20)
('Mr.', 'System Administrator', 'admin@university.edu', '+1-555-0001', '$2b$10$hashedpassword20', 3, 3, '/uploads/profiles/admin.jpg', NULL, 'REGISTERED', TRUE, '2024-06-03 18:00:00', FALSE);

-- ===========================================
-- STEP 3: Verify the data insertion
-- ===========================================

-- Check total count
SELECT 'Total AppUser records' as metric, COUNT(*) as count FROM AppUser;

-- Check distribution by role
SELECT 
    r.Name as Role,
    COUNT(u.id) as UserCount
FROM AppUser u
JOIN Role r ON u.RoleID = r.id
GROUP BY r.Name
ORDER BY UserCount DESC;

-- Check distribution by status
SELECT 
    Status,
    COUNT(*) as UserCount
FROM AppUser
GROUP BY Status
ORDER BY UserCount DESC;

-- Check distribution by specialization
SELECT 
    s.Name as Specialization,
    COUNT(u.id) as UserCount
FROM AppUser u
LEFT JOIN Specialization s ON u.SpecializationID = s.id
GROUP BY s.Name
ORDER BY UserCount DESC;

-- Display all users with their related information
SELECT 
    u.id,
    u.Salutation,
    u.Name,
    u.Email,
    u.MobileNumber,
    r.Name as Role,
    a.Name as Avatar,
    s.Name as Specialization,
    u.Status,
    u.Active,
    u.isPopupFlag,
    u.LastLoginDateTime
FROM AppUser u
LEFT JOIN Role r ON u.RoleID = r.id
LEFT JOIN Avatar a ON u.AvatarID = a.id
LEFT JOIN Specialization s ON u.SpecializationID = s.id
ORDER BY u.id;

-- Show users who haven't logged in recently
SELECT 
    u.Name,
    u.Email,
    u.Status,
    u.LastLoginDateTime,
    CASE 
        WHEN u.LastLoginDateTime IS NULL THEN 'Never logged in'
        WHEN u.LastLoginDateTime < DATE_SUB(NOW(), INTERVAL 7 DAY) THEN 'Inactive (>7 days)'
        ELSE 'Recently active'
    END as LoginStatus
FROM AppUser u
WHERE u.Active = TRUE
ORDER BY u.LastLoginDateTime DESC;

-- ===========================================
-- DATA INSERTION COMPLETE
-- ===========================================

-- NOTES:
-- 1. 20 diverse users have been inserted covering different roles and statuses
-- 2. Email addresses are unique as required by the table constraint
-- 3. Mobile numbers follow a consistent format
-- 4. PasswordHash values are sample bcrypt hashes (not real passwords)
-- 5. Users have different statuses: SEEDED, PROFILEVERIFIED, DISPLAYIMAGEUPLOADED, REGISTERED
-- 6. Mix of active and inactive users, with and without profile pictures
-- 7. Variety of specializations and roles represented
-- 8. Some users have recent login times, others don't
-- 9. isPopupFlag is set to TRUE for some users for testing purposes
-- 10. All foreign key relationships are properly maintained