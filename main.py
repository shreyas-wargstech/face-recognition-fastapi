# main.py - LMS Face Recognition API
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, LargeBinary, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

import face_recognition
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

# Initialize FastAPI app
app = FastAPI(
    title="LMS Face Recognition API",
    description="Face registration and verification system for LMS quiz authentication",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./lms_face.db")

# Handle SQLite vs PostgreSQL connection settings
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, echo=True, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL, echo=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Security
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Encryption for face embeddings
ENCRYPTION_KEY = os.getenv("FACE_ENCRYPTION_KEY", Fernet.generate_key())
cipher_suite = Fernet(ENCRYPTION_KEY)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database Models
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
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, index=True)
    face_embedding = Column(LargeBinary)  # Encrypted face features
    registration_photo_path = Column(String)
    quality_score = Column(Float)
    face_confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class FaceVerification(Base):
    __tablename__ = "face_verifications"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, index=True)
    quiz_id = Column(String, nullable=True)  # Optional quiz identifier
    verification_photo_path = Column(String)
    similarity_score = Column(Float)
    verification_result = Column(Boolean)
    threshold_used = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    ip_address = Column(String)
    user_agent = Column(Text)

# Create tables (comment out for production - use init_db.py instead)
# Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Face Recognition Service
class FaceRecognitionService:
    def __init__(self):
        self.similarity_threshold = 0.6  # Configurable threshold
        self.min_face_size = 50  # Minimum face size in pixels
        
    def extract_face_encoding(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Extract face encoding from image"""
        try:
            # Convert to RGB if needed
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image_array
            
            # Find face locations
            face_locations = face_recognition.face_locations(rgb_image, model="hog")
            
            if not face_locations:
                return {
                    "success": False,
                    "error": "No face detected in the image",
                    "face_count": 0
                }
            
            if len(face_locations) > 1:
                return {
                    "success": False,
                    "error": "Multiple faces detected. Please ensure only one face is visible",
                    "face_count": len(face_locations)
                }
            
            # Extract face encoding
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            if not face_encodings:
                return {
                    "success": False,
                    "error": "Could not extract facial features",
                    "face_count": len(face_locations)
                }
            
            # Calculate face quality metrics
            face_location = face_locations[0]
            quality_score = self._calculate_face_quality(rgb_image, face_location)
            
            return {
                "success": True,
                "encoding": face_encodings[0],
                "face_location": face_location,
                "quality_score": quality_score,
                "face_count": 1
            }
            
        except Exception as e:
            logger.error(f"Error extracting face encoding: {str(e)}")
            return {
                "success": False,
                "error": f"Processing error: {str(e)}",
                "face_count": 0
            }
    
    def compare_faces(self, registered_encoding: np.ndarray, current_encoding: np.ndarray) -> Dict[str, Any]:
        """Compare two face encodings"""
        try:
            # Calculate face distance (lower is better)
            face_distance = face_recognition.face_distance([registered_encoding], current_encoding)[0]
            
            # Convert distance to similarity score (0-100)
            similarity_score = max(0, (1 - face_distance) * 100)
            
            # Determine if faces match
            is_match = face_distance <= self.similarity_threshold
            
            return {
                "similarity_score": float(similarity_score),
                "is_match": bool(is_match),  # Convert numpy.bool_ to Python bool
                "face_distance": float(face_distance),
                "threshold_used": float(self.similarity_threshold)
            }
            
        except Exception as e:
            logger.error(f"Error comparing faces: {str(e)}")
            return {
                "similarity_score": 0.0,
                "is_match": False,
                "error": str(e)
            }
    
    def _calculate_face_quality(self, image: np.ndarray, face_location: tuple) -> float:
        """Calculate face quality score based on various factors"""
        top, right, bottom, left = face_location
        face_image = image[top:bottom, left:right]
        
        # Convert to grayscale for analysis
        gray_face = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        
        # Calculate sharpness using Laplacian variance
        sharpness = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        
        # Calculate brightness
        brightness = np.mean(gray_face)
        
        # Face size factor
        face_area = (bottom - top) * (right - left)
        size_factor = min(1.0, face_area / (100 * 100))  # Normalize to 100x100 minimum
        
        # Combine factors into quality score (0-100)
        quality_score = min(100, (
            (sharpness / 100) * 40 +  # Sharpness weight: 40%
            (brightness / 255) * 30 +  # Brightness weight: 30%
            size_factor * 30           # Size weight: 30%
        ))
        
        return max(0, quality_score)

# Initialize service
face_service = FaceRecognitionService()

# Helper functions
def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
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

# Helper functions
def encrypt_face_encoding(encoding: np.ndarray) -> bytes:
    """Encrypt face encoding for secure storage"""
    encoding_bytes = pickle.dumps(encoding)
    return cipher_suite.encrypt(encoding_bytes)

def decrypt_face_encoding(encrypted_data: bytes) -> np.ndarray:
    """Decrypt face encoding from storage"""
    decrypted_bytes = cipher_suite.decrypt(encrypted_data)
    return pickle.loads(decrypted_bytes)

def save_uploaded_file(file: UploadFile, user_id: int, file_type: str) -> str:
    """Save uploaded file and return path"""
    # Create upload directory
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    # Generate unique filename
    file_extension = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
    filename = f"{user_id}_{file_type}_{uuid.uuid4().hex}.{file_extension}"
    file_path = os.path.join(upload_dir, filename)
    
    # Save file
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())
    
    return file_path

# Authentication middleware (simplified)
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if token != "lms-face-token-123":  # Replace with proper JWT verification
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return token

# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "LMS Face Recognition API",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/v1/face/register")
async def register_face(
    user_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    token: str = Depends(verify_token)
):
    """
    Register a user's face for future verification
    
    - **user_id**: The LMS user ID
    - **file**: Face photo (JPG/PNG)
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Check if user already has active registration
        existing_registration = db.query(FaceRegistration).filter(
            FaceRegistration.user_id == user_id,
            FaceRegistration.is_active == True
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
        if result["quality_score"] < 30:  # Minimum quality threshold
            raise HTTPException(
                status_code=400, 
                detail=f"Photo quality too low ({result['quality_score']:.1f}/100). Please upload a clearer photo."
            )
        
        # Save photo file
        file.file.seek(0)  # Reset file pointer
        photo_path = save_uploaded_file(file, user_id, "registration")
        
        # Encrypt and store face encoding
        encrypted_encoding = encrypt_face_encoding(result["encoding"])
        
        # Deactivate existing registration if any
        if existing_registration:
            existing_registration.is_active = False
            db.commit()
        
        # Create new registration
        registration = FaceRegistration(
            user_id=user_id,
            face_embedding=encrypted_encoding,
            registration_photo_path=photo_path,
            quality_score=result["quality_score"],
            face_confidence=95.0,  # Based on successful extraction
        )
        
        db.add(registration)
        db.commit()
        db.refresh(registration)
        
        logger.info(f"Face registered successfully for user {user_id}")
        
        response = {
            "success": True,
            "registration_id": registration.id,
            "user_id": user_id,
            "quality_score": result["quality_score"],
            "message": "Face registered successfully"
        }
        
        # Convert any numpy types to Python native types
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
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    token: str = Depends(verify_token)
):
    """
    Verify a user's identity by comparing with registered face
    
    - **user_id**: The LMS user ID
    - **quiz_id**: Optional quiz identifier for audit trail
    - **file**: Current face photo for verification
    """
    try:
        # Get user's registered face
        registration = db.query(FaceRegistration).filter(
            FaceRegistration.user_id == user_id,
            FaceRegistration.is_active == True
        ).first()
        
        if not registration:
            raise HTTPException(
                status_code=404, 
                detail="No face registration found for this user. Please register first."
            )
        
        # Read and process current image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image)
        
        # Extract face encoding from current photo
        current_result = face_service.extract_face_encoding(image_array)
        
        if not current_result["success"]:
            # Log failed verification attempt
            file.file.seek(0)
            photo_path = save_uploaded_file(file, user_id, "verification_failed")
            
            verification = FaceVerification(
                user_id=user_id,
                quiz_id=quiz_id,
                verification_photo_path=photo_path,
                similarity_score=0.0,
                verification_result=False,
                threshold_used=face_service.similarity_threshold
            )
            db.add(verification)
            db.commit()
            
            raise HTTPException(status_code=400, detail=current_result["error"])
        
        # Decrypt registered face encoding
        registered_encoding = decrypt_face_encoding(registration.face_embedding)
        
        # Compare faces
        comparison_result = face_service.compare_faces(
            registered_encoding, 
            current_result["encoding"]
        )
        
        # Save verification photo
        file.file.seek(0)
        photo_path = save_uploaded_file(file, user_id, "verification")
        
        # Log verification attempt
        verification = FaceVerification(
            user_id=user_id,
            quiz_id=quiz_id,
            verification_photo_path=photo_path,
            similarity_score=comparison_result["similarity_score"],
            verification_result=comparison_result["is_match"],
            threshold_used=comparison_result["threshold_used"]
        )
        
        db.add(verification)
        db.commit()
        db.refresh(verification)
        
        # Prepare response
        response = {
            "success": True,
            "verification_id": verification.id,
            "user_id": user_id,
            "quiz_id": quiz_id,
            "verified": comparison_result["is_match"],
            "similarity_score": comparison_result["similarity_score"],
            "threshold": comparison_result["threshold_used"],
            "quality_score": current_result["quality_score"],
            "message": "Identity verified successfully" if comparison_result["is_match"] else "Identity verification failed"
        }
        
        # Convert any numpy types to Python native types
        response = convert_numpy_types(response)
        
        if comparison_result["is_match"]:
            logger.info(f"Face verification PASSED for user {user_id} (similarity: {comparison_result['similarity_score']:.1f}%)")
        else:
            logger.warning(f"Face verification FAILED for user {user_id} (similarity: {comparison_result['similarity_score']:.1f}%)")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error verifying face for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")

@app.get("/api/v1/face/registration/{user_id}")
async def get_registration_status(
    user_id: int,
    db: Session = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Get face registration status for a user"""
    registration = db.query(FaceRegistration).filter(
        FaceRegistration.user_id == user_id,
        FaceRegistration.is_active == True
    ).first()
    
    if registration:
        return {
            "registered": True,
            "registration_id": registration.id,
            "quality_score": registration.quality_score,
            "registered_at": registration.created_at.isoformat()
        }
    else:
        return {"registered": False}

@app.get("/api/v1/face/verifications/{user_id}")
async def get_verification_history(
    user_id: int,
    limit: int = 10,
    db: Session = Depends(get_db),
    token: str = Depends(verify_token)
):
    """Get verification history for a user"""
    verifications = db.query(FaceVerification).filter(
        FaceVerification.user_id == user_id
    ).order_by(FaceVerification.created_at.desc()).limit(limit).all()
    
    return {
        "user_id": user_id,
        "verifications": [
            {
                "verification_id": v.id,
                "quiz_id": v.quiz_id,
                "verified": v.verification_result,
                "similarity_score": v.similarity_score,
                "verified_at": v.created_at.isoformat()
            }
            for v in verifications
        ]
    }

@app.get("/api/v1/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "face_recognition": "operational",
            "database": "operational",
            "api": "operational"
        },
        "version": "1.0.0",
        "threshold": face_service.similarity_threshold
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)