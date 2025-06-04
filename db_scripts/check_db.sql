-- MySQL Verification Script
-- Run this script AFTER creating base tables AND running the DeepFace migration
-- This will verify that all tables and structures are properly created

-- ===========================================
-- VERIFICATION QUERIES
-- ===========================================

-- 1. Check all tables exist
SELECT 
    TABLE_NAME,
    TABLE_ROWS,
    DATA_LENGTH,
    CREATE_TIME
FROM information_schema.TABLES 
WHERE TABLE_SCHEMA = DATABASE() 
AND TABLE_NAME IN ('AppUser', 'Face', 'Face_backup', 'FaceVerification', 'FaceSession')
ORDER BY TABLE_NAME;

-- 2. Verify Face table has all new columns after migration
SELECT 
    COLUMN_NAME,
    DATA_TYPE,
    IS_NULLABLE,
    COLUMN_DEFAULT,
    COLUMN_COMMENT
FROM information_schema.COLUMNS 
WHERE TABLE_SCHEMA = DATABASE() 
AND TABLE_NAME = 'Face'
ORDER BY ORDINAL_POSITION;

-- 3. Check Face table data and new columns
SELECT 
    id,
    UserID,
    CASE WHEN FaceEmbedding IS NULL THEN 'NULL' ELSE 'HAS_DATA' END as FaceEmbedding_status,
    ModelName,
    DetectorBackend,
    QualityScore,
    FaceConfidence,
    IsActive,
    RegistrationSource,
    CreationDateTime
FROM Face 
LIMIT 5;

-- 4. Verify indexes exist
SELECT 
    TABLE_NAME,
    INDEX_NAME,
    COLUMN_NAME,
    NON_UNIQUE
FROM information_schema.STATISTICS 
WHERE TABLE_SCHEMA = DATABASE() 
AND TABLE_NAME IN ('Face', 'FaceVerification', 'FaceSession')
ORDER BY TABLE_NAME, INDEX_NAME, SEQ_IN_INDEX;

-- 5. Check views exist and work
SELECT * FROM v_face_registration_stats LIMIT 5;
SELECT * FROM v_user_face_status LIMIT 5;

-- 6. Test stored procedures exist
SELECT 
    ROUTINE_NAME,
    ROUTINE_TYPE,
    CREATED,
    LAST_ALTERED
FROM information_schema.ROUTINES 
WHERE ROUTINE_SCHEMA = DATABASE() 
AND ROUTINE_TYPE = 'PROCEDURE';

-- 7. Test one stored procedure
CALL GetSystemHealthMetrics();

-- 8. Verify foreign key constraints
SELECT 
    CONSTRAINT_NAME,
    TABLE_NAME,
    COLUMN_NAME,
    REFERENCED_TABLE_NAME,
    REFERENCED_COLUMN_NAME
FROM information_schema.KEY_COLUMN_USAGE 
WHERE TABLE_SCHEMA = DATABASE() 
AND REFERENCED_TABLE_NAME IS NOT NULL;

-- 9. Check triggers exist
SELECT 
    TRIGGER_NAME,
    EVENT_MANIPULATION,
    EVENT_OBJECT_TABLE,
    ACTION_TIMING
FROM information_schema.TRIGGERS 
WHERE TRIGGER_SCHEMA = DATABASE();

-- 10. Sample data verification
SELECT 
    'AppUser' as table_name,
    COUNT(*) as total_records,
    SUM(CASE WHEN Active = TRUE THEN 1 ELSE 0 END) as active_records
FROM AppUser

UNION ALL

SELECT 
    'Face' as table_name,
    COUNT(*) as total_records,
    SUM(CASE WHEN IsActive = TRUE THEN 1 ELSE 0 END) as active_records
FROM Face

UNION ALL

SELECT 
    'FaceVerification' as table_name,
    COUNT(*) as total_records,
    SUM(CASE WHEN VerificationResult = TRUE THEN 1 ELSE 0 END) as successful_records
FROM FaceVerification

UNION ALL

SELECT 
    'FaceSession' as table_name,
    COUNT(*) as total_records,
    SUM(CASE WHEN IsActive = TRUE THEN 1 ELSE 0 END) as active_records
FROM FaceSession;

-- ===========================================
-- EXPECTED RESULTS SUMMARY
-- ===========================================

/*
Expected Results After Successful Migration:

1. Tables Created:
   - AppUser (existing)
   - Face (modified with new columns)
   - Face_backup (backup of original Face table)
   - FaceVerification (new)
   - FaceSession (new)

2. New Face Table Columns:
   - FaceEmbedding (LONGBLOB)
   - ModelName (VARCHAR)
   - DetectorBackend (VARCHAR)
   - QualityScore (FLOAT)
   - FaceConfidence (FLOAT)
   - ImagePath (VARCHAR)
   - ImageBase64 (TEXT)
   - IsActive (BOOLEAN)
   - RegistrationSource (VARCHAR)

3. Views Created:
   - v_face_registration_stats
   - v_verification_stats
   - v_user_face_status

4. Stored Procedures:
   - GetUserVerificationSummary
   - CleanupOldVerifications
   - GetSystemHealthMetrics

5. Triggers:
   - face_update_trigger

6. Indexes:
   - Multiple performance indexes on all tables

If any of these are missing, check the migration script execution logs.
*/