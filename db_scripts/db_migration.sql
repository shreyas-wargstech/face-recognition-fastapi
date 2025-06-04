-- MySQL Migration Script for DeepFace Integration
-- This script modifies your existing Face table and creates new tables for enhanced face recognition

-- ===========================================
-- STEP 1: Backup existing Face table
-- ===========================================
CREATE TABLE Face_backup AS SELECT * FROM Face;

-- ===========================================
-- STEP 2: Add new columns to existing Face table
-- ===========================================

-- Add new columns for DeepFace integration
ALTER TABLE Face 
ADD COLUMN FaceEmbedding LONGBLOB COMMENT 'Encrypted face embedding from DeepFace',
ADD COLUMN ModelName VARCHAR(50) DEFAULT 'ArcFace' COMMENT 'DeepFace model used',
ADD COLUMN DetectorBackend VARCHAR(50) DEFAULT 'retinaface' COMMENT 'Face detector backend',
ADD COLUMN QualityScore FLOAT COMMENT 'Face image quality score (0-100)',
ADD COLUMN FaceConfidence FLOAT COMMENT 'Face detection confidence score',
ADD COLUMN ImagePath VARCHAR(500) COMMENT 'Path to stored face image',
ADD COLUMN ImageBase64 TEXT COMMENT 'Base64 encoded image for small images',
ADD COLUMN IsActive BOOLEAN DEFAULT TRUE COMMENT 'Whether this face registration is active',
ADD COLUMN RegistrationSource VARCHAR(50) DEFAULT 'api' COMMENT 'Source of registration (api, mobile, web)';

-- Add index for better performance
CREATE INDEX idx_face_userid_active ON Face(UserID, IsActive);
CREATE INDEX idx_face_active ON Face(IsActive);

-- ===========================================
-- STEP 3: Create FaceVerification table for audit trail
-- ===========================================

CREATE TABLE FaceVerification (
    id INT PRIMARY KEY AUTO_INCREMENT,
    UserID INT NOT NULL,
    QuizID VARCHAR(100) COMMENT 'Quiz identifier',
    CourseID VARCHAR(100) COMMENT 'Course identifier',
    
    -- Verification results
    VerificationResult BOOLEAN NOT NULL COMMENT 'Whether verification passed',
    SimilarityScore FLOAT NOT NULL COMMENT 'Similarity score (0-100)',
    Distance FLOAT NOT NULL COMMENT 'Face distance metric',
    ThresholdUsed FLOAT NOT NULL COMMENT 'Threshold used for verification',
    
    -- Technical details
    ModelName VARCHAR(50) NOT NULL COMMENT 'DeepFace model used',
    DistanceMetric VARCHAR(20) NOT NULL COMMENT 'Distance metric used',
    ProcessingTime FLOAT COMMENT 'Processing time in milliseconds',
    
    -- Image data
    VerificationImagePath VARCHAR(500) COMMENT 'Path to verification image',
    QualityScore FLOAT COMMENT 'Image quality score',
    
    -- Metadata
    IPAddress VARCHAR(45) COMMENT 'User IP address',
    UserAgent TEXT COMMENT 'User agent string',
    VerificationDateTime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CreationDateTime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes
    INDEX idx_verification_userid (UserID),
    INDEX idx_verification_quiz (QuizID),
    INDEX idx_verification_course (CourseID),
    INDEX idx_verification_result (VerificationResult),
    INDEX idx_verification_datetime (VerificationDateTime),
    
    -- Foreign key constraint (if you have foreign key constraints enabled)
    FOREIGN KEY (UserID) REFERENCES AppUser(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ===========================================
-- STEP 4: Create FaceSession table for session management (optional)
-- ===========================================

CREATE TABLE FaceSession (
    id INT PRIMARY KEY AUTO_INCREMENT,
    UserID INT NOT NULL,
    SessionToken VARCHAR(255) NOT NULL UNIQUE,
    QuizID VARCHAR(100),
    CourseID VARCHAR(100),
    
    -- Session details
    VerificationID INT COMMENT 'Reference to successful verification',
    ExpiresAt TIMESTAMP NOT NULL,
    IsActive BOOLEAN DEFAULT TRUE,
    
    -- Metadata
    IPAddress VARCHAR(45),
    UserAgent TEXT,
    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    LastAccessedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Indexes
    INDEX idx_session_userid (UserID),
    INDEX idx_session_token (SessionToken),
    INDEX idx_session_active (IsActive),
    INDEX idx_session_expires (ExpiresAt),
    
    -- Foreign key constraints
    FOREIGN KEY (UserID) REFERENCES AppUser(id) ON DELETE CASCADE,
    FOREIGN KEY (VerificationID) REFERENCES FaceVerification(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ===========================================
-- STEP 5: Create performance monitoring views
-- ===========================================

-- View for face registration statistics
CREATE VIEW v_face_registration_stats AS
SELECT 
    COUNT(DISTINCT f.UserID) as total_registered_users,
    COUNT(*) as total_face_registrations,
    AVG(f.QualityScore) as avg_quality_score,
    AVG(f.FaceConfidence) as avg_face_confidence,
    f.ModelName,
    f.DetectorBackend,
    DATE(f.CreationDateTime) as registration_date
FROM Face f 
WHERE f.IsActive = TRUE
GROUP BY f.ModelName, f.DetectorBackend, DATE(f.CreationDateTime);

-- View for verification statistics
CREATE VIEW v_verification_stats AS
SELECT 
    DATE(v.VerificationDateTime) as verification_date,
    COUNT(*) as total_verifications,
    SUM(CASE WHEN v.VerificationResult = TRUE THEN 1 ELSE 0 END) as successful_verifications,
    ROUND(SUM(CASE WHEN v.VerificationResult = TRUE THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as success_rate,
    AVG(v.SimilarityScore) as avg_similarity_score,
    AVG(v.ProcessingTime) as avg_processing_time,
    v.ModelName,
    v.DistanceMetric
FROM FaceVerification v 
GROUP BY DATE(v.VerificationDateTime), v.ModelName, v.DistanceMetric;

-- View for user face status
CREATE VIEW v_user_face_status AS
SELECT 
    u.id as user_id,
    u.Name as user_name,
    u.Email as user_email,
    u.Status as user_status,
    u.Active as user_active,
    CASE WHEN f.id IS NOT NULL THEN TRUE ELSE FALSE END as has_face_registered,
    f.id as face_id,
    f.QualityScore as face_quality,
    f.ModelName as face_model,
    f.CreationDateTime as face_registered_at,
    (
        SELECT COUNT(*) 
        FROM FaceVerification fv 
        WHERE fv.UserID = u.id 
        AND fv.VerificationDateTime >= DATE_SUB(NOW(), INTERVAL 30 DAY)
    ) as verifications_last_30_days,
    (
        SELECT COUNT(*) 
        FROM FaceVerification fv 
        WHERE fv.UserID = u.id 
        AND fv.VerificationResult = TRUE 
        AND fv.VerificationDateTime >= DATE_SUB(NOW(), INTERVAL 30 DAY)
    ) as successful_verifications_last_30_days
FROM AppUser u
LEFT JOIN Face f ON u.id = f.UserID AND f.IsActive = TRUE;

-- ===========================================
-- STEP 6: Create stored procedures for common operations
-- ===========================================

DELIMITER //

-- Procedure to get user verification summary
CREATE PROCEDURE GetUserVerificationSummary(IN userId INT)
BEGIN
    SELECT 
        u.id,
        u.Name,
        u.Email,
        CASE WHEN f.id IS NOT NULL THEN 'Registered' ELSE 'Not Registered' END as face_status,
        f.QualityScore as face_quality,
        f.ModelName as face_model,
        COUNT(fv.id) as total_verifications,
        SUM(CASE WHEN fv.VerificationResult = TRUE THEN 1 ELSE 0 END) as successful_verifications,
        CASE 
            WHEN COUNT(fv.id) > 0 THEN 
                ROUND(SUM(CASE WHEN fv.VerificationResult = TRUE THEN 1 ELSE 0 END) * 100.0 / COUNT(fv.id), 2)
            ELSE 0 
        END as success_rate,
        MAX(fv.VerificationDateTime) as last_verification
    FROM AppUser u
    LEFT JOIN Face f ON u.id = f.UserID AND f.IsActive = TRUE
    LEFT JOIN FaceVerification fv ON u.id = fv.UserID
    WHERE u.id = userId
    GROUP BY u.id;
END //

-- Procedure to clean up old verification records
CREATE PROCEDURE CleanupOldVerifications(IN daysToKeep INT)
BEGIN
    DECLARE rowsDeleted INT DEFAULT 0;
    
    DELETE FROM FaceVerification 
    WHERE VerificationDateTime < DATE_SUB(NOW(), INTERVAL daysToKeep DAY);
    
    SET rowsDeleted = ROW_COUNT();
    
    SELECT CONCAT('Deleted ', rowsDeleted, ' verification records older than ', daysToKeep, ' days') as result;
END //

-- Procedure to get system health metrics
CREATE PROCEDURE GetSystemHealthMetrics()
BEGIN
    SELECT 
        'Users' as metric_type,
        COUNT(*) as total_count,
        SUM(CASE WHEN Active = TRUE THEN 1 ELSE 0 END) as active_count
    FROM AppUser
    
    UNION ALL
    
    SELECT 
        'Face Registrations' as metric_type,
        COUNT(*) as total_count,
        SUM(CASE WHEN IsActive = TRUE THEN 1 ELSE 0 END) as active_count
    FROM Face
    
    UNION ALL
    
    SELECT 
        'Verifications (Last 24h)' as metric_type,
        COUNT(*) as total_count,
        SUM(CASE WHEN VerificationResult = TRUE THEN 1 ELSE 0 END) as successful_count
    FROM FaceVerification 
    WHERE VerificationDateTime >= DATE_SUB(NOW(), INTERVAL 24 HOUR);
END //

DELIMITER ;

-- ===========================================
-- STEP 7: Create triggers for audit logging
-- ===========================================

-- Trigger to log face registration changes
DELIMITER //
CREATE TRIGGER face_update_trigger 
AFTER UPDATE ON Face
FOR EACH ROW
BEGIN
    IF OLD.IsActive = TRUE AND NEW.IsActive = FALSE THEN
        INSERT INTO FaceVerification (
            UserID, 
            VerificationResult, 
            SimilarityScore, 
            Distance, 
            ThresholdUsed, 
            ModelName, 
            DistanceMetric,
            ProcessingTime
        ) VALUES (
            NEW.UserID, 
            FALSE, 
            0, 
            999, 
            0, 
            'SYSTEM', 
            'deactivation',
            0
        );
    END IF;
END //
DELIMITER ;

-- ===========================================
-- STEP 8: Sample data and testing queries
-- ===========================================

-- Test query to check face registration status
-- SELECT 
--     u.id,
--     u.Name,
--     u.Email,
--     CASE WHEN f.id IS NOT NULL THEN 'Yes' ELSE 'No' END as face_registered,
--     f.QualityScore,
--     f.ModelName
-- FROM AppUser u
-- LEFT JOIN Face f ON u.id = f.UserID AND f.IsActive = TRUE
-- WHERE u.Active = TRUE
-- LIMIT 10;

-- Test query to check recent verifications
-- SELECT 
--     v.*,
--     u.Name as user_name
-- FROM FaceVerification v
-- JOIN AppUser u ON v.UserID = u.id
-- ORDER BY v.VerificationDateTime DESC
-- LIMIT 10;

-- ===========================================
-- STEP 9: Performance optimization indexes
-- ===========================================

-- Composite indexes for common queries
CREATE INDEX idx_face_user_model_active ON Face(UserID, ModelName, IsActive);
CREATE INDEX idx_verification_user_date ON FaceVerification(UserID, VerificationDateTime);
CREATE INDEX idx_verification_result_date ON FaceVerification(VerificationResult, VerificationDateTime);

-- ===========================================
-- STEP 10: Data migration for existing records
-- ===========================================

-- Update existing Face records with default values
UPDATE Face 
SET 
    ModelName = 'migration_legacy',
    DetectorBackend = 'legacy',
    IsActive = TRUE,
    RegistrationSource = 'legacy'
WHERE ModelName IS NULL;

-- ===========================================
-- MIGRATION COMPLETE
-- ===========================================

-- Verify migration
SELECT 
    'Face table columns' as check_type,
    COUNT(*) as total_records,
    SUM(CASE WHEN FaceEmbedding IS NOT NULL THEN 1 ELSE 0 END) as with_embeddings,
    SUM(CASE WHEN IsActive = TRUE THEN 1 ELSE 0 END) as active_records
FROM Face

UNION ALL

SELECT 
    'FaceVerification table' as check_type,
    COUNT(*) as total_records,
    0 as with_embeddings,
    SUM(CASE WHEN VerificationResult = TRUE THEN 1 ELSE 0 END) as successful_verifications
FROM FaceVerification;

-- Show table structures
-- SHOW CREATE TABLE Face;
-- SHOW CREATE TABLE FaceVerification;

-- Note: After running this migration:
-- 1. Test the new API endpoints
-- 2. Gradually migrate existing face data to use DeepFace embeddings
-- 3. Monitor performance and adjust indexes as needed
-- 4. Set up regular cleanup jobs for old verification records