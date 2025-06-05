# main.py - LMS Face Recognition API integrated with existing MySQL database and AWS S3
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, LargeBinary, Text, TIMESTAMP, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func

# DeepFace imports
from deepface import DeepFace
import cv2
import numpy as np
import base64
import io
from PIL import Image
import uuid
import os
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import json
from passlib.context import CryptContext
import secrets
from cryptography.fernet import Fernet
import pickle
import pymysql
from dotenv import load_dotenv

# Import S3 service
from s3_service import S3Service

load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="LMS Face Recognition API",
    description="Face registration and verification system integrated with existing LMS MySQL database and AWS S3",
    version="2.1.0"
)

# Configure CORS for your Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js development
        "http://localhost:3001",  # Alternative Next.js port
        "https://your-domain.com",  # Your production domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration for MySQL
DATABASE_URL = os.getenv("DATABASE_URL")

# Create engine for MySQL
engine = create_engine(
    DATABASE_URL,
    echo=True,
    pool_pre_ping=True,
    pool_recycle=300
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Security
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Encryption for face embeddings
ENCRYPTION_KEY = os.getenv("FACE_ENCRYPTION_KEY", Fernet.generate_key())
cipher_suite = Fernet(ENCRYPTION_KEY)

# Initialize S3 service
try:
    s3_service = S3Service()
    logger.info("✅ S3 service initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize S3 service: {str(e)}")
    s3_service = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database Models - Updated with S3 fields
class AppUser(Base):
    """Model matching your existing AppUser table"""
    __tablename__ = "AppUser"
    
    id = Column(Integer, primary_key=True, index=True)
    Salutation = Column(String(10), nullable=True)
    Name = Column(String(255), nullable=False)
    Status = Column(String(50), nullable=False, default='SEEDED')
    Email = Column(String(128), nullable=True)
    MobileNumber = Column(String(16), nullable=True)
    PasswordHash = Column(String(255), nullable=True)
    OTPHash = Column(String(255), nullable=True)
    OTPDateTime = Column(DateTime, nullable=True)
    RoleID = Column(Integer, nullable=False)
    AvatarID = Column(Integer, nullable=True)
    ProfilePictureUrl = Column(String(255), nullable=True)
    SpecializationID = Column(Integer, nullable=True)
    Active = Column(Boolean, default=True)
    LastLoginDateTime = Column(DateTime, nullable=True)
    isPopupFlag = Column(Boolean, default=False)
    CreationDateTime = Column(TIMESTAMP, server_default=func.now())
    UpdationDateTime = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

class Face(Base):
    """Enhanced Face model for DeepFace integration with S3 storage"""
    __tablename__ = "Face"
    
    id = Column(Integer, primary_key=True, index=True)
    UserID = Column(Integer, nullable=False, index=True)
    
    # Enhanced face data storage
    FaceEmbedding = Column(LargeBinary, nullable=True)  # Encrypted face embedding
    FaceData = Column(String(255), nullable=True)  # Keep for backward compatibility
    
    # DeepFace specific fields
    ModelName = Column(String(50), default="ArcFace")
    DetectorBackend = Column(String(50), default="retinaface")
    QualityScore = Column(Float, nullable=True)
    FaceConfidence = Column(Float, nullable=True)
    
    # S3 storage fields
    S3Key = Column(String(500), nullable=True)  # S3 object key
    S3Url = Column(String(1000), nullable=True)  # S3 object URL
    ImagePath = Column(String(500), nullable=True)  # Keep for backward compatibility
    ImageBase64 = Column(Text, nullable=True)  # For small images (deprecated)
    
    # Status and metadata
    IsActive = Column(Boolean, default=True)
    RegistrationSource = Column(String(50), default="api")  # api, mobile, web
    StorageType = Column(String(20), default="s3")  # s3, local, base64
    
    CreationDateTime = Column(TIMESTAMP, server_default=func.now())
    UpdateDateTime = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

class FaceVerification(Base):
    """Enhanced verification table with S3 storage"""
    __tablename__ = "FaceVerification"
    
    id = Column(Integer, primary_key=True, index=True)
    UserID = Column(Integer, nullable=False, index=True)
    QuizID = Column(String(100), nullable=True)  # Your quiz identifier
    CourseID = Column(String(100), nullable=True)  # Course identifier
    
    # Verification results
    VerificationResult = Column(Boolean, nullable=False)
    SimilarityScore = Column(Float, nullable=False)
    Distance = Column(Float, nullable=False)
    ThresholdUsed = Column(Float, nullable=False)
    
    # Technical details
    ModelName = Column(String(50), nullable=False)
    DistanceMetric = Column(String(20), nullable=False)
    ProcessingTime = Column(Float, nullable=True)  # in milliseconds
    
    # S3 storage fields
    S3Key = Column(String(500), nullable=True)  # S3 object key
    S3Url = Column(String(1000), nullable=True)  # S3 object URL
    VerificationImagePath = Column(String(500), nullable=True)  # Keep for backward compatibility
    QualityScore = Column(Float, nullable=True)
    StorageType = Column(String(20), default="s3")  # s3, local
    
    # Metadata
    IPAddress = Column(String(45), nullable=True)
    UserAgent = Column(Text, nullable=True)
    VerificationDateTime = Column(TIMESTAMP, server_default=func.now())
    
    CreationDateTime = Column(TIMESTAMP, server_default=func.now())

# Create tables
# Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# DeepFace Recognition Service (same as before)
class DeepFaceRecognitionService:
    def __init__(self):
        self.model_name = os.getenv("DEEPFACE_MODEL", "ArcFace")
        self.detector_backend = os.getenv("DEEPFACE_DETECTOR", "retinaface")
        self.distance_metric = os.getenv("DEEPFACE_DISTANCE", "cosine")
        self.enforce_detection = True
        self.align = True
        self.anti_spoofing = False
        
        # Optimized thresholds for different models
        self.thresholds = {
            "VGG-Face": {"cosine": 0.68, "euclidean": 1.17, "euclidean_l2": 1.17},
            "Facenet": {"cosine": 0.40, "euclidean": 10, "euclidean_l2": 0.80},
            "Facenet512": {"cosine": 0.30, "euclidean": 23.56, "euclidean_l2": 1.04},
            "ArcFace": {"cosine": 0.68, "euclidean": 4.15, "euclidean_l2": 1.13},
            "Dlib": {"cosine": 0.07, "euclidean": 0.6, "euclidean_l2": 0.4},
        }
        
        self.min_quality_score = float(os.getenv("MIN_FACE_QUALITY_SCORE", "30.0"))
        self.min_face_confidence = float(os.getenv("MIN_FACE_CONFIDENCE", "0.7"))
        
    def extract_face_encoding(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Extract face encoding using DeepFace - optimized for speed"""
        import time
        start_time = time.time()
        
        try:
            # Convert image format if needed
            if len(image_array.shape) == 3:
                if image_array.shape[2] == 4:  # RGBA
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
                elif image_array.shape[2] == 3:  # BGR
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            
            # Extract faces with quality check
            faces = DeepFace.extract_faces(
                img_path=image_array,
                detector_backend=self.detector_backend,
                enforce_detection=self.enforce_detection,
                align=self.align,
                anti_spoofing=self.anti_spoofing
            )
            
            if not faces:
                return {
                    "success": False,
                    "error": "No face detected in the image",
                    "face_count": 0,
                    "processing_time": (time.time() - start_time) * 1000
                }
            
            if len(faces) > 1:
                return {
                    "success": False,
                    "error": "Multiple faces detected. Please ensure only one face is visible",
                    "face_count": len(faces),
                    "processing_time": (time.time() - start_time) * 1000
                }
            
            face_data = faces[0]
            face_confidence = face_data.get('confidence', 0)
            
            if face_confidence < self.min_face_confidence:
                return {
                    "success": False,
                    "error": f"Face detection confidence too low ({face_confidence:.2f}). Please use a clearer image",
                    "face_count": 1,
                    "processing_time": (time.time() - start_time) * 1000
                }
            
            # Extract face embedding
            embeddings = DeepFace.represent(
                img_path=image_array,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=self.enforce_detection,
                align=self.align,
                anti_spoofing=self.anti_spoofing
            )
            
            if not embeddings:
                return {
                    "success": False,
                    "error": "Could not extract facial features",
                    "face_count": 1,
                    "processing_time": (time.time() - start_time) * 1000
                }
            
            embedding = embeddings[0]['embedding']
            facial_area = embeddings[0]['facial_area']
            
            # Calculate quality score
            quality_score = self._calculate_face_quality(image_array, facial_area, face_confidence)
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "success": True,
                "encoding": np.array(embedding),
                "facial_area": facial_area,
                "quality_score": quality_score,
                "face_confidence": face_confidence,
                "face_count": 1,
                "model_name": self.model_name,
                "processing_time": processing_time
            }
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Error extracting face encoding: {str(e)}")
            return {
                "success": False,
                "error": f"Processing error: {str(e)}",
                "face_count": 0,
                "processing_time": processing_time
            }
    
    def compare_faces(self, registered_encoding: np.ndarray, current_encoding: np.ndarray) -> Dict[str, Any]:
        """Compare two face encodings with performance metrics"""
        import time
        start_time = time.time()
        
        try:
            # Calculate distance based on metric
            if self.distance_metric == "cosine":
                a = np.matmul(np.transpose(registered_encoding), current_encoding)
                b = np.sum(np.multiply(registered_encoding, registered_encoding))
                c = np.sum(np.multiply(current_encoding, current_encoding))
                distance = 1 - (a / (np.sqrt(b) * np.sqrt(c)))
            elif self.distance_metric == "euclidean":
                distance = np.linalg.norm(registered_encoding - current_encoding)
            elif self.distance_metric == "euclidean_l2":
                registered_norm = registered_encoding / np.linalg.norm(registered_encoding)
                current_norm = current_encoding / np.linalg.norm(current_encoding)
                distance = np.linalg.norm(registered_norm - current_norm)
            else:
                raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
            
            # Get threshold
            threshold = self.thresholds.get(self.model_name, {}).get(self.distance_metric, 0.68)
            
            # Calculate similarity score
            if self.distance_metric == "cosine":
                similarity_score = max(0, (1 - distance) * 100)
            else:
                similarity_score = max(0, (1 - min(distance / threshold, 1)) * 100)
            
            is_match = distance <= threshold
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "similarity_score": float(similarity_score),
                "is_match": bool(is_match),
                "distance": float(distance),
                "threshold_used": float(threshold),
                "distance_metric": self.distance_metric,
                "model_name": self.model_name,
                "processing_time": processing_time
            }
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Error comparing faces: {str(e)}")
            return {
                "similarity_score": 0.0,
                "is_match": False,
                "distance": float('inf'),
                "error": str(e),
                "processing_time": processing_time
            }
    
    def _calculate_face_quality(self, image: np.ndarray, facial_area: dict, face_confidence: float) -> float:
        """Calculate face quality score"""
        try:
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            face_image = image[y:y+h, x:x+w]
            
            if face_image.size == 0:
                return 0.0
            
            # Convert to grayscale
            if len(face_image.shape) == 3:
                gray_face = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
            else:
                gray_face = face_image
            
            # Calculate metrics
            sharpness = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            brightness = np.mean(gray_face)
            face_area = w * h
            size_factor = min(1.0, face_area / (100 * 100))
            
            # Combined quality score
            quality_score = (
                (min(sharpness / 100, 1.0)) * 25 +
                (brightness / 255) * 20 +
                size_factor * 25 +
                face_confidence * 30
            ) * 100
            
            return max(0, min(100, quality_score))
            
        except Exception as e:
            logger.warning(f"Error calculating face quality: {str(e)}")
            return 50.0

# Initialize service
face_service = DeepFaceRecognitionService()

# Utility functions
def encrypt_face_encoding(encoding: np.ndarray) -> bytes:
    """Encrypt face encoding for secure storage"""
    encoding_bytes = pickle.dumps(encoding)
    return cipher_suite.encrypt(encoding_bytes)

def decrypt_face_encoding(encrypted_data: bytes) -> np.ndarray:
    """Decrypt face encoding from storage"""
    decrypted_bytes = cipher_suite.decrypt(encrypted_data)
    return pickle.loads(decrypted_bytes)

def save_image_to_storage(file: UploadFile, user_id: int, image_type: str) -> Dict[str, str]:
    """
    Save image to S3 or local storage (fallback)
    Returns dict with storage info
    """
    if s3_service:
        try:
            s3_key, s3_url = s3_service.upload_face_image(file, user_id, image_type)
            return {
                "storage_type": "s3",
                "s3_key": s3_key,
                "s3_url": s3_url,
                "path": s3_url  # For backward compatibility
            }
        except Exception as e:
            logger.error(f"S3 upload failed, falling back to local storage: {str(e)}")
    
    # Fallback to local storage
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    file_extension = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
    filename = f"{user_id}_{image_type}_{uuid.uuid4().hex}.{file_extension}"
    file_path = os.path.join(upload_dir, filename)
    
    file.file.seek(0)
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())
    
    return {
        "storage_type": "local",
        "s3_key": None,
        "s3_url": None,
        "path": file_path
    }

def convert_numpy_types(obj):
    """Convert numpy types to Python native types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

# Authentication middleware
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    # TODO: Implement proper JWT verification matching your Node.js backend
    if token != "lms-face-token-123":
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return token

# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "LMS Face Recognition API",
        "version": "2.1.0",
        "engine": "DeepFace",
        "database": "MySQL",
        "storage": "AWS S3" if s3_service else "Local",
        "model": face_service.model_name,
        "detector": face_service.detector_backend,
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/v1/face/register")
async def register_face(
    user_id: int = Form(...),
    file: UploadFile = File(...),
    source: str = Form("api"),  # api, mobile, web
    db: Session = Depends(get_db)
):
    """
    Register a user's face for future verification
    
    - **user_id**: The LMS user ID (from AppUser table)
    - **file**: Face photo (JPG/PNG)
    - **source**: Registration source (api, mobile, web)
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Check if user exists in AppUser table
        user = db.query(AppUser).filter(AppUser.id == user_id, AppUser.Active == True).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found or inactive")
        
        # Check existing registration
        existing_face = db.query(Face).filter(
            Face.UserID == user_id,
            Face.IsActive == True
        ).first()
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image)
        
        # Extract face encoding
        result = face_service.extract_face_encoding(image_array)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Check quality threshold
        if result["quality_score"] < face_service.min_quality_score:
            raise HTTPException(
                status_code=400,
                detail=f"Photo quality too low ({result['quality_score']:.1f}/100). Please upload a clearer photo."
            )
        
        # Save image to S3 or local storage
        file.file.seek(0)
        storage_info = save_image_to_storage(file, user_id, "registration")
        
        # Encrypt face embedding
        encrypted_embedding = encrypt_face_encoding(result["encoding"])
        
        # Deactivate existing registration if any
        if existing_face:
            existing_face.IsActive = False
            db.commit()
        
        # Create new face registration
        new_face = Face(
            UserID=user_id,
            FaceEmbedding=encrypted_embedding,
            FaceData=f"deepface_{result['model_name']}_{uuid.uuid4().hex[:8]}",  # Backward compatibility
            ModelName=result["model_name"],
            DetectorBackend=face_service.detector_backend,
            QualityScore=result["quality_score"],
            FaceConfidence=result["face_confidence"],
            S3Key=storage_info["s3_key"],
            S3Url=storage_info["s3_url"],
            ImagePath=storage_info["path"],  # Backward compatibility
            StorageType=storage_info["storage_type"],
            RegistrationSource=source
        )
        
        db.add(new_face)
        db.commit()
        db.refresh(new_face)
        
        logger.info(f"Face registered for user {user_id} using {result['model_name']} (quality: {result['quality_score']:.1f}) - Storage: {storage_info['storage_type']}")
        
        response = {
            "success": True,
            "face_id": new_face.id,
            "user_id": user_id,
            "user_name": user.Name,
            "quality_score": result["quality_score"],
            "face_confidence": result["face_confidence"],
            "model_name": result["model_name"],
            "processing_time": result["processing_time"],
            "storage_type": storage_info["storage_type"],
            "s3_url": storage_info["s3_url"] if storage_info["storage_type"] == "s3" else None,
            "message": "Face registered successfully"
        }
        
        return convert_numpy_types(response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering face for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/api/v1/face/verify")
async def verify_face(
    user_id: int = Form(...),
    quiz_id: str = Form(None),
    course_id: str = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Verify a user's identity for quiz/course access
    
    - **user_id**: The LMS user ID
    - **quiz_id**: Quiz identifier
    - **course_id**: Course identifier
    - **file**: Current face photo for verification
    """
    try:
        # Get user and face registration
        user = db.query(AppUser).filter(AppUser.id == user_id, AppUser.Active == True).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found or inactive")
        
        face_registration = db.query(Face).filter(
            Face.UserID == user_id,
            Face.IsActive == True
        ).first()
        
        if not face_registration:
            raise HTTPException(
                status_code=404,
                detail="No face registration found. Please register your face first."
            )
        
        # Process current image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image)
        
        # Extract current face encoding
        current_result = face_service.extract_face_encoding(image_array)
        
        if not current_result["success"]:
            # Save failed verification image and log
            file.file.seek(0)
            storage_info = save_image_to_storage(file, user_id, "verification_failed")
            
            verification = FaceVerification(
                UserID=user_id,
                QuizID=quiz_id,
                CourseID=course_id,
                VerificationResult=False,
                SimilarityScore=0.0,
                Distance=float('inf'),
                ThresholdUsed=0.0,
                ModelName=face_service.model_name,
                DistanceMetric=face_service.distance_metric,
                ProcessingTime=current_result.get("processing_time", 0),
                S3Key=storage_info["s3_key"],
                S3Url=storage_info["s3_url"],
                VerificationImagePath=storage_info["path"],
                StorageType=storage_info["storage_type"],
                QualityScore=0.0
            )
            db.add(verification)
            db.commit()
            
            raise HTTPException(status_code=400, detail=current_result["error"])
        
        # Decrypt registered face encoding
        registered_encoding = decrypt_face_encoding(face_registration.FaceEmbedding)
        
        # Compare faces
        comparison_result = face_service.compare_faces(
            registered_encoding,
            current_result["encoding"]
        )
        
        # Save verification image
        file.file.seek(0)
        storage_info = save_image_to_storage(file, user_id, "verification")
        
        # Log verification attempt
        verification = FaceVerification(
            UserID=user_id,
            QuizID=quiz_id,
            CourseID=course_id,
            VerificationResult=comparison_result["is_match"],
            SimilarityScore=comparison_result["similarity_score"],
            Distance=comparison_result["distance"],
            ThresholdUsed=comparison_result["threshold_used"],
            ModelName=comparison_result["model_name"],
            DistanceMetric=comparison_result["distance_metric"],
            ProcessingTime=comparison_result["processing_time"],
            S3Key=storage_info["s3_key"],
            S3Url=storage_info["s3_url"],
            VerificationImagePath=storage_info["path"],
            StorageType=storage_info["storage_type"],
            QualityScore=current_result["quality_score"]
        )
        
        db.add(verification)
        db.commit()
        db.refresh(verification)
        
        # Update user's last login if verification successful
        if comparison_result["is_match"]:
            user.LastLoginDateTime = datetime.utcnow()
            db.commit()
        
        response = {
            "success": True,
            "verification_id": verification.id,
            "user_id": user_id,
            "user_name": user.Name,
            "quiz_id": quiz_id,
            "course_id": course_id,
            "verified": comparison_result["is_match"],
            "similarity_score": comparison_result["similarity_score"],
            "distance": comparison_result["distance"],
            "threshold": comparison_result["threshold_used"],
            "quality_score": current_result["quality_score"],
            "processing_time": comparison_result["processing_time"],
            "model_name": comparison_result["model_name"],
            "storage_type": storage_info["storage_type"],
            "message": "Identity verified successfully" if comparison_result["is_match"] else "Identity verification failed"
        }
        
        if comparison_result["is_match"]:
            logger.info(f"✅ Face verification PASSED for user {user_id} ({user.Name}) - Similarity: {comparison_result['similarity_score']:.1f}%")
        else:
            logger.warning(f"❌ Face verification FAILED for user {user_id} ({user.Name}) - Similarity: {comparison_result['similarity_score']:.1f}%")
        
        return convert_numpy_types(response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error verifying face for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")

@app.get("/api/v1/face/status/{user_id}")
async def get_face_status(
    user_id: int,
    db: Session = Depends(get_db)
):
    """Get face registration status for a user"""
    try:
        # Get user
        user = db.query(AppUser).filter(AppUser.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get face registration
        face = db.query(Face).filter(
            Face.UserID == user_id,
            Face.IsActive == True
        ).first()
        
        if face:
            # Generate presigned URL if S3 is used
            presigned_url = None
            if face.S3Key and s3_service:
                try:
                    presigned_url = s3_service.get_presigned_url(face.S3Key, expiration=3600)
                except Exception as e:
                    logger.warning(f"Failed to generate presigned URL: {str(e)}")
            
            return {
                "user_id": user_id,
                "user_name": user.Name,
                "registered": True,
                "face_id": face.id,
                "quality_score": face.QualityScore,
                "face_confidence": face.FaceConfidence,
                "model_name": face.ModelName,
                "detector_backend": face.DetectorBackend,
                "registration_source": face.RegistrationSource,
                "storage_type": face.StorageType,
                "s3_url": face.S3Url,
                "presigned_url": presigned_url,
                "registered_at": face.CreationDateTime.isoformat() if face.CreationDateTime else None
            }
        else:
            return {
                "user_id": user_id,
                "user_name": user.Name,
                "registered": False
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting face status for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get face status")

@app.get("/api/v1/face/verifications/{user_id}")
async def get_verification_history(
    user_id: int,
    limit: int = 10,
    quiz_id: str = None,
    course_id: str = None,
    db: Session = Depends(get_db)
):
    """Get verification history for a user"""
    try:
        query = db.query(FaceVerification).filter(FaceVerification.UserID == user_id)
        
        if quiz_id:
            query = query.filter(FaceVerification.QuizID == quiz_id)
        if course_id:
            query = query.filter(FaceVerification.CourseID == course_id)
        
        verifications = query.order_by(
            FaceVerification.VerificationDateTime.desc()
        ).limit(limit).all()
        
        # Get user name
        user = db.query(AppUser).filter(AppUser.id == user_id).first()
        user_name = user.Name if user else "Unknown"
        
        verification_list = []
        for v in verifications:
            # Generate presigned URL if S3 is used
            presigned_url = None
            if v.S3Key and s3_service:
                try:
                    presigned_url = s3_service.get_presigned_url(v.S3Key, expiration=3600)
                except Exception as e:
                    logger.warning(f"Failed to generate presigned URL: {str(e)}")
            
            verification_list.append({
                "verification_id": v.id,
                "quiz_id": v.QuizID,
                "course_id": v.CourseID,
                "verified": v.VerificationResult,
                "similarity_score": v.SimilarityScore,
                "distance": v.Distance,
                "quality_score": v.QualityScore,
                "processing_time": v.ProcessingTime,
                "model_name": v.ModelName,
                "distance_metric": v.DistanceMetric,
                "storage_type": v.StorageType,
                "s3_url": v.S3Url,
                "presigned_url": presigned_url,
                "verified_at": v.VerificationDateTime.isoformat() if v.VerificationDateTime else None
            })
        
        return {
            "user_id": user_id,
            "user_name": user_name,
            "total_verifications": len(verifications),
            "verifications": verification_list
        }
        
    except Exception as e:
        logger.error(f"Error getting verification history for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get verification history")

@app.get("/api/v1/health")
async def health_check(db: Session = Depends(get_db)):
    """Comprehensive health check"""
    try:
        # Test database connection
        db.execute(text("SELECT 1"))
        db_status = "operational"
    except:
        db_status = "error"
    
    # Test S3 connection
    s3_status = "not_configured"
    s3_info = {}
    if s3_service:
        s3_info = s3_service.check_bucket_health()
        s3_status = s3_info.get("status", "error")
    
    return {
        "status": "healthy" if db_status == "operational" and s3_status in ["healthy", "not_configured"] else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "engine": "DeepFace",
        "database": "MySQL",
        "storage": "AWS S3" if s3_service else "Local",
        "services": {
            "deepface": "operational",
            "database": db_status,
            "s3": s3_status,
            "api": "operational"
        },
        "configuration": {
            "model": face_service.model_name,
            "detector": face_service.detector_backend,
            "distance_metric": face_service.distance_metric,
            "anti_spoofing": face_service.anti_spoofing,
            "threshold": face_service.thresholds.get(face_service.model_name, {}).get(face_service.distance_metric, "N/A")
        },
        "s3_info": s3_info,
        "performance": {
            "min_quality_score": face_service.min_quality_score,
            "min_face_confidence": face_service.min_face_confidence
        },
        "version": "2.1.0"
    }

@app.get("/api/v1/stats")
async def get_system_stats(db: Session = Depends(get_db)):
    """Get system statistics"""
    try:
        # Count registered users
        total_users = db.query(AppUser).filter(AppUser.Active == True).count()
        registered_faces = db.query(Face).filter(Face.IsActive == True).count()
        
        # Count by storage type
        s3_faces = db.query(Face).filter(Face.IsActive == True, Face.StorageType == "s3").count()
        local_faces = db.query(Face).filter(Face.IsActive == True, Face.StorageType == "local").count()
        
        # Count verifications in last 24 hours
        from datetime import timedelta
        yesterday = datetime.utcnow() - timedelta(days=1)
        recent_verifications = db.query(FaceVerification).filter(
            FaceVerification.VerificationDateTime >= yesterday
        ).count()
        
        successful_verifications = db.query(FaceVerification).filter(
            FaceVerification.VerificationDateTime >= yesterday,
            FaceVerification.VerificationResult == True
        ).count()
        
        success_rate = (successful_verifications / recent_verifications * 100) if recent_verifications > 0 else 0
        
        return {
            "total_users": total_users,
            "registered_faces": registered_faces,
            "registration_rate": (registered_faces / total_users * 100) if total_users > 0 else 0,
            "storage_distribution": {
                "s3": s3_faces,
                "local": local_faces
            },
            "recent_verifications_24h": recent_verifications,
            "success_rate_24h": success_rate,
            "system_health": "excellent" if success_rate > 95 else "good" if success_rate > 85 else "needs_attention"
        }
        
    except Exception as e:
        logger.error(f"Error getting system stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get system statistics")

@app.post("/api/v1/admin/cleanup")
async def cleanup_old_images(
    days: int = 30,
    db: Session = Depends(get_db)
):
    """Clean up old verification images from S3"""
    if not s3_service:
        raise HTTPException(status_code=503, detail="S3 service not available")
    
    try:
        deleted_count = s3_service.cleanup_old_images(days)
        return {
            "success": True,
            "deleted_images": deleted_count,
            "days_threshold": days,
            "message": f"Cleaned up {deleted_count} old verification images"
        }
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        raise HTTPException(status_code=500, detail="Cleanup failed")

@app.get("/api/v1/admin/s3/status")
async def get_s3_status():
    """Get S3 bucket status and statistics"""
    if not s3_service:
        raise HTTPException(status_code=503, detail="S3 service not configured")
    
    try:
        bucket_info = s3_service.check_bucket_health()
        return bucket_info
    except Exception as e:
        logger.error(f"Error getting S3 status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get S3 status")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)