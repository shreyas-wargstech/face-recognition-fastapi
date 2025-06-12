# main.py - Enhanced LMS Face Recognition API with DeepFace Streaming and Anti-Spoofing
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
import asyncio
import time
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor
import pickle
from cryptography.fernet import Fernet
from dotenv import load_dotenv

load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="LMS Face Recognition API with Real-Time Streaming",
    description="Real-time face registration and verification system with anti-spoofing",
    version="3.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://your-domain.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL, echo=True, pool_pre_ping=True, pool_recycle=300)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Encryption for face embeddings
ENCRYPTION_KEY = os.getenv("FACE_ENCRYPTION_KEY", Fernet.generate_key())
cipher_suite = Fernet(ENCRYPTION_KEY)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database Models (keeping existing models)
class AppUser(Base):
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
    __tablename__ = "Face"
    
    id = Column(Integer, primary_key=True, index=True)
    UserID = Column(Integer, nullable=False, index=True)
    FaceEmbedding = Column(LargeBinary, nullable=True)
    FaceData = Column(String(255), nullable=True)
    ModelName = Column(String(50), default="ArcFace")
    DetectorBackend = Column(String(50), default="retinaface")
    QualityScore = Column(Float, nullable=True)
    FaceConfidence = Column(Float, nullable=True)
    S3Key = Column(String(500), nullable=True)
    S3Url = Column(String(1000), nullable=True)
    ImagePath = Column(String(500), nullable=True)
    ImageBase64 = Column(Text, nullable=True)
    IsActive = Column(Boolean, default=True)
    RegistrationSource = Column(String(50), default="api")
    StorageType = Column(String(20), default="s3")
    CreationDateTime = Column(TIMESTAMP, server_default=func.now())
    UpdateDateTime = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

class FaceVerification(Base):
    __tablename__ = "FaceVerification"
    
    id = Column(Integer, primary_key=True, index=True)
    UserID = Column(Integer, nullable=False, index=True)
    QuizID = Column(String(100), nullable=True)
    CourseID = Column(String(100), nullable=True)
    VerificationResult = Column(Boolean, nullable=False)
    SimilarityScore = Column(Float, nullable=False)
    Distance = Column(Float, nullable=False)
    ThresholdUsed = Column(Float, nullable=False)
    ModelName = Column(String(50), nullable=False)
    DistanceMetric = Column(String(20), nullable=False)
    ProcessingTime = Column(Float, nullable=True)
    S3Key = Column(String(500), nullable=True)
    S3Url = Column(String(1000), nullable=True)
    VerificationImagePath = Column(String(500), nullable=True)
    QualityScore = Column(Float, nullable=True)
    StorageType = Column(String(20), default="s3")
    IPAddress = Column(String(45), nullable=True)
    UserAgent = Column(Text, nullable=True)
    VerificationDateTime = Column(TIMESTAMP, server_default=func.now())
    CreationDateTime = Column(TIMESTAMP, server_default=func.now())

# New table for streaming sessions
class StreamingSession(Base):
    __tablename__ = "StreamingSession"
    
    id = Column(Integer, primary_key=True, index=True)
    SessionID = Column(String(100), nullable=False, unique=True, index=True)
    UserID = Column(Integer, nullable=False)
    SessionType = Column(String(50), nullable=False)  # 'registration' or 'verification'
    Status = Column(String(50), nullable=False, default='active')  # 'active', 'completed', 'failed'
    FramesProcessed = Column(Integer, default=0)
    LivenessScore = Column(Float, nullable=True)
    AntiSpoofingScore = Column(Float, nullable=True)
    QualityScore = Column(Float, nullable=True)
    StartTime = Column(TIMESTAMP, server_default=func.now())
    EndTime = Column(TIMESTAMP, nullable=True)
    CreationDateTime = Column(TIMESTAMP, server_default=func.now())

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Real-time Face Recognition Service with Anti-Spoofing
class RealTimeFaceRecognitionService:
    def __init__(self):
        self.model_name = os.getenv("DEEPFACE_MODEL", "ArcFace")
        self.detector_backend = os.getenv("DEEPFACE_DETECTOR", "retinaface")
        self.distance_metric = os.getenv("DEEPFACE_DISTANCE", "cosine")
        self.anti_spoofing = True
        self.enforce_detection = True
        self.align = True
        
        # Quality thresholds
        self.min_quality_score = float(os.getenv("MIN_FACE_QUALITY_SCORE", "50.0"))
        self.min_face_confidence = float(os.getenv("MIN_FACE_CONFIDENCE", "0.8"))
        self.liveness_threshold = float(os.getenv("LIVENESS_THRESHOLD", "0.7"))
        
        # Frame processing settings
        self.required_frames = 10  # Minimum frames for registration/verification
        self.max_frames = 30       # Maximum frames to process
        self.frame_skip = 2        # Process every nth frame for performance
        
        # Thread pool for heavy processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"✅ RealTimeFaceRecognitionService initialized with {self.model_name}")

    def extract_face_with_antispoofing(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Extract face with anti-spoofing detection"""
        try:
            start_time = time.time()
            
            # Extract faces with anti-spoofing
            faces = DeepFace.extract_faces(
                img_path=image_array,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=self.align,
                anti_spoofing=self.anti_spoofing
            )
            
            if not faces:
                return {
                    "success": False,
                    "error": "No face detected",
                    "spoofing_detected": False,
                    "processing_time": (time.time() - start_time) * 1000
                }
            
            # Check for spoofing in any detected face
            for face in faces:
                if not face.get('is_real', True):
                    return {
                        "success": False,
                        "error": "Spoofing detected - please use your real face",
                        "spoofing_detected": True,
                        "antispoofing_score": face.get('antispoof_score', 0.0),
                        "processing_time": (time.time() - start_time) * 1000
                    }
            
            # Use the best quality face
            best_face = max(faces, key=lambda x: x.get('confidence', 0))
            
            # Extract face embedding
            embeddings = DeepFace.represent(
                img_path=image_array,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=self.align,
                anti_spoofing=self.anti_spoofing
            )
            
            if not embeddings:
                return {
                    "success": False,
                    "error": "Could not extract facial features",
                    "spoofing_detected": False,
                    "processing_time": (time.time() - start_time) * 1000
                }
            
            embedding = embeddings[0]['embedding']
            facial_area = embeddings[0]['facial_area']
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(image_array, facial_area, best_face.get('confidence', 0.9))
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "success": True,
                "encoding": np.array(embedding),
                "facial_area": facial_area,
                "quality_score": quality_score,
                "face_confidence": best_face.get('confidence', 0.9),
                "antispoofing_score": best_face.get('antispoof_score', 1.0),
                "is_real": best_face.get('is_real', True),
                "spoofing_detected": False,
                "model_name": self.model_name,
                "processing_time": processing_time
            }
            
        except ValueError as e:
            if "Spoof detected" in str(e):
                return {
                    "success": False,
                    "error": "Spoofing attack detected",
                    "spoofing_detected": True,
                    "processing_time": (time.time() - start_time) * 1000
                }
            else:
                return {
                    "success": False,
                    "error": str(e),
                    "spoofing_detected": False,
                    "processing_time": (time.time() - start_time) * 1000
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Processing error: {str(e)}",
                "spoofing_detected": False,
                "processing_time": (time.time() - start_time) * 1000
            }

    def _calculate_quality_score(self, image: np.ndarray, facial_area: dict, face_confidence: float) -> float:
        """Calculate comprehensive quality score"""
        try:
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            face_image = image[y:y+h, x:x+w]
            
            if face_image.size == 0:
                return 0.0
            
            # Convert to grayscale for analysis
            if len(face_image.shape) == 3:
                gray_face = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
            else:
                gray_face = face_image
            
            # Sharpness (Laplacian variance)
            sharpness = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            
            # Brightness
            brightness = np.mean(gray_face)
            
            # Contrast
            contrast = gray_face.std()
            
            # Face size factor
            face_area = w * h
            size_factor = min(1.0, face_area / (150 * 150))  # Optimal size 150x150
            
            # Combined quality score
            quality_score = (
                (min(sharpness / 150, 1.0)) * 30 +      # Sharpness weight
                (1 - abs(brightness - 128) / 128) * 20 + # Optimal brightness around 128
                (min(contrast / 50, 1.0)) * 20 +        # Contrast weight
                size_factor * 10 +                       # Size weight
                face_confidence * 20                     # Detection confidence
            ) * 100
            
            return max(0, min(100, quality_score))
            
        except Exception as e:
            logger.warning(f"Error calculating quality score: {str(e)}")
            return 50.0

    def compare_faces_with_verification(self, registered_encoding: np.ndarray, current_encoding: np.ndarray) -> Dict[str, Any]:
        """Compare faces with detailed verification metrics"""
        try:
            start_time = time.time()
            
            # Calculate cosine similarity
            similarity = np.dot(registered_encoding, current_encoding) / (
                np.linalg.norm(registered_encoding) * np.linalg.norm(current_encoding)
            )
            
            # Calculate distance
            distance = 1 - similarity
            
            # Thresholds for ArcFace model
            threshold = 0.68 if self.model_name == "ArcFace" else 0.40
            
            # Calculate similarity percentage
            similarity_score = max(0, similarity * 100)
            
            # Verification result
            is_match = distance <= threshold and similarity_score >= 70  # Additional threshold
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "similarity_score": float(similarity_score),
                "is_match": bool(is_match),
                "distance": float(distance),
                "threshold_used": float(threshold),
                "confidence": float(similarity_score / 100),
                "model_name": self.model_name,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error comparing faces: {str(e)}")
            return {
                "similarity_score": 0.0,
                "is_match": False,
                "distance": float('inf'),
                "threshold_used": 0.68,
                "confidence": 0.0,
                "error": str(e),
                "processing_time": 0
            }

# Initialize service
face_service = RealTimeFaceRecognitionService()

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Dict] = {}
        
    async def connect(self, websocket: WebSocket, session_id: str, user_id: int, session_type: str):
        await websocket.accept()
        self.active_connections[session_id] = {
            "websocket": websocket,
            "user_id": user_id,
            "session_type": session_type,
            "frame_buffer": deque(maxlen=50),
            "frame_count": 0,
            "quality_scores": [],
            "liveness_scores": [],
            "start_time": time.time(),
            "last_processed": 0
        }
        logger.info(f"🔗 WebSocket connected: {session_id} for user {user_id} ({session_type})")
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"❌ WebSocket disconnected: {session_id}")
    
    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]["websocket"]
            await websocket.send_text(json.dumps(message))

manager = ConnectionManager()

# Utility functions
def encrypt_face_encoding(encoding: np.ndarray) -> bytes:
    """Encrypt face encoding for secure storage"""
    encoding_bytes = pickle.dumps(encoding)
    return cipher_suite.encrypt(encoding_bytes)

def decrypt_face_encoding(encrypted_data: bytes) -> np.ndarray:
    """Decrypt face encoding from storage"""
    decrypted_bytes = cipher_suite.decrypt(encrypted_data)
    return pickle.loads(decrypted_bytes)

def decode_base64_frame(frame_data: str) -> np.ndarray:
    """Decode base64 frame to numpy array"""
    try:
        # Remove data URL prefix if present
        if ',' in frame_data:
            frame_data = frame_data.split(',')[1]
        
        # Decode base64
        img_data = base64.b64decode(frame_data)
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(img_data))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Convert RGBA to RGB if necessary
        if len(image_array.shape) == 3 and image_array.shape[2] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        elif len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        
        return image_array
    except Exception as e:
        logger.error(f"Error decoding frame: {str(e)}")
        return None

# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "LMS Face Recognition API with Real-Time Streaming",
        "version": "3.0.0",
        "engine": "DeepFace",
        "features": ["Real-time streaming", "Anti-spoofing", "Liveness detection"],
        "model": face_service.model_name,
        "detector": face_service.detector_backend,
        "anti_spoofing": face_service.anti_spoofing,
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.websocket("/ws/face-registration/{user_id}")
async def face_registration_stream(websocket: WebSocket, user_id: int, db: Session = Depends(get_db)):
    """Real-time face registration with anti-spoofing"""
    session_id = f"reg_{user_id}_{uuid.uuid4().hex[:8]}"
    
    try:
        # Check if user exists
        user = db.query(AppUser).filter(AppUser.id == user_id, AppUser.Active == True).first()
        if not user:
            await websocket.close(code=4004, reason="User not found")
            return
        
        # Create streaming session record
        streaming_session = StreamingSession(
            SessionID=session_id,
            UserID=user_id,
            SessionType="registration"
        )
        db.add(streaming_session)
        db.commit()
        
        await manager.connect(websocket, session_id, user_id, "registration")
        
        # Send initial status
        await manager.send_message(session_id, {
            "type": "connected",
            "session_id": session_id,
            "user_id": user_id,
            "user_name": user.Name,
            "required_frames": face_service.required_frames,
            "message": "Ready for face registration. Please look at the camera."
        })
        
        best_frames = []
        quality_scores = []
        antispoofing_scores = []
        
        while True:
            # Receive frame data
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "frame":
                connection = manager.active_connections[session_id]
                frame_data = message_data["frame"]
                
                # Decode frame
                frame = decode_base64_frame(frame_data)
                if frame is None:
                    await manager.send_message(session_id, {
                        "type": "error",
                        "message": "Invalid frame data"
                    })
                    continue
                
                # Skip frames for performance
                connection["frame_count"] += 1
                if connection["frame_count"] % face_service.frame_skip != 0:
                    continue
                
                # Process frame
                result = face_service.extract_face_with_antispoofing(frame)
                
                # Update session
                streaming_session.FramesProcessed += 1
                
                if result["spoofing_detected"]:
                    streaming_session.Status = "failed"
                    db.commit()
                    
                    await manager.send_message(session_id, {
                        "type": "spoofing_detected",
                        "message": result["error"],
                        "antispoofing_score": result.get("antispoofing_score", 0.0)
                    })
                    continue
                
                if not result["success"]:
                    await manager.send_message(session_id, {
                        "type": "frame_processed",
                        "success": False,
                        "message": result["error"],
                        "frames_collected": len(best_frames),
                        "required_frames": face_service.required_frames
                    })
                    continue
                
                # Check quality
                if result["quality_score"] < face_service.min_quality_score:
                    await manager.send_message(session_id, {
                        "type": "frame_processed",
                        "success": False,
                        "message": f"Frame quality too low ({result['quality_score']:.1f}/100)",
                        "quality_score": result["quality_score"],
                        "frames_collected": len(best_frames),
                        "required_frames": face_service.required_frames
                    })
                    continue
                
                # Store good frame
                best_frames.append({
                    "encoding": result["encoding"],
                    "quality_score": result["quality_score"],
                    "antispoofing_score": result["antispoofing_score"],
                    "face_confidence": result["face_confidence"]
                })
                
                quality_scores.append(result["quality_score"])
                antispoofing_scores.append(result["antispoofing_score"])
                
                await manager.send_message(session_id, {
                    "type": "frame_processed",
                    "success": True,
                    "quality_score": result["quality_score"],
                    "antispoofing_score": result["antispoofing_score"],
                    "face_confidence": result["face_confidence"],
                    "frames_collected": len(best_frames),
                    "required_frames": face_service.required_frames,
                    "message": f"Good frame captured! ({len(best_frames)}/{face_service.required_frames})"
                })
                
                # Check if we have enough frames
                if len(best_frames) >= face_service.required_frames:
                    # Select best frame
                    best_frame = max(best_frames, key=lambda x: x["quality_score"])
                    
                    # Calculate average scores
                    avg_quality = sum(quality_scores) / len(quality_scores)
                    avg_antispoofing = sum(antispoofing_scores) / len(antispoofing_scores)
                    
                    # Encrypt and store face encoding
                    encrypted_embedding = encrypt_face_encoding(best_frame["encoding"])
                    
                    # Deactivate existing registration
                    existing_face = db.query(Face).filter(
                        Face.UserID == user_id,
                        Face.IsActive == True
                    ).first()
                    
                    if existing_face:
                        existing_face.IsActive = False
                    
                    # Create new face registration
                    new_face = Face(
                        UserID=user_id,
                        FaceEmbedding=encrypted_embedding,
                        FaceData=f"stream_{face_service.model_name}_{uuid.uuid4().hex[:8]}",
                        ModelName=face_service.model_name,
                        DetectorBackend=face_service.detector_backend,
                        QualityScore=avg_quality,
                        FaceConfidence=best_frame["face_confidence"],
                        RegistrationSource="stream"
                    )
                    
                    db.add(new_face)
                    
                    # Update streaming session
                    streaming_session.Status = "completed"
                    streaming_session.QualityScore = avg_quality
                    streaming_session.AntiSpoofingScore = avg_antispoofing
                    streaming_session.EndTime = datetime.utcnow()
                    
                    db.commit()
                    db.refresh(new_face)
                    
                    await manager.send_message(session_id, {
                        "type": "registration_complete",
                        "success": True,
                        "face_id": new_face.id,
                        "user_name": user.Name,
                        "quality_score": avg_quality,
                        "antispoofing_score": avg_antispoofing,
                        "frames_processed": len(best_frames),
                        "message": "Face registration completed successfully!"
                    })
                    
                    logger.info(f"✅ Face registered successfully for user {user_id} via streaming")
                    break
                
                # Stop if we've processed too many frames
                if len(best_frames) >= face_service.max_frames:
                    break
            
            elif message_data.get("type") == "stop":
                break
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Error in face registration stream: {str(e)}")
        try:
            await manager.send_message(session_id, {
                "type": "error",
                "message": f"Registration failed: {str(e)}"
            })
        except:
            pass
    finally:
        manager.disconnect(session_id)

@app.websocket("/ws/face-verification/{user_id}")
async def face_verification_stream(
    websocket: WebSocket, 
    user_id: int, 
    quiz_id: str = None, 
    course_id: str = None,
    db: Session = Depends(get_db)
):
    """Real-time face verification with anti-spoofing"""
    session_id = f"ver_{user_id}_{uuid.uuid4().hex[:8]}"
    
    try:
        # Check if user exists and has registered face
        user = db.query(AppUser).filter(AppUser.id == user_id, AppUser.Active == True).first()
        if not user:
            await websocket.close(code=4004, reason="User not found")
            return
        
        face_registration = db.query(Face).filter(
            Face.UserID == user_id,
            Face.IsActive == True
        ).first()
        
        if not face_registration:
            await websocket.close(code=4003, reason="No face registration found")
            return
        
        # Create streaming session record
        streaming_session = StreamingSession(
            SessionID=session_id,
            UserID=user_id,
            SessionType="verification"
        )
        db.add(streaming_session)
        db.commit()
        
        await manager.connect(websocket, session_id, user_id, "verification")
        
        # Decrypt registered face encoding
        registered_encoding = decrypt_face_encoding(face_registration.FaceEmbedding)
        
        # Send initial status
        await manager.send_message(session_id, {
            "type": "connected",
            "session_id": session_id,
            "user_id": user_id,
            "user_name": user.Name,
            "quiz_id": quiz_id,
            "course_id": course_id,
            "required_frames": face_service.required_frames,
            "message": "Ready for face verification. Please look at the camera."
        })
        
        verification_frames = []
        quality_scores = []
        antispoofing_scores = []
        similarity_scores = []
        
        while True:
            # Receive frame data
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "frame":
                connection = manager.active_connections[session_id]
                frame_data = message_data["frame"]
                
                # Decode frame
                frame = decode_base64_frame(frame_data)
                if frame is None:
                    await manager.send_message(session_id, {
                        "type": "error",
                        "message": "Invalid frame data"
                    })
                    continue
                
                # Skip frames for performance
                connection["frame_count"] += 1
                if connection["frame_count"] % face_service.frame_skip != 0:
                    continue
                
                # Process frame
                result = face_service.extract_face_with_antispoofing(frame)
                
                # Update session
                streaming_session.FramesProcessed += 1
                
                if result["spoofing_detected"]:
                    streaming_session.Status = "failed"
                    db.commit()
                    
                    await manager.send_message(session_id, {
                        "type": "spoofing_detected",
                        "message": result["error"],
                        "antispoofing_score": result.get("antispoofing_score", 0.0)
                    })
                    continue
                
                if not result["success"]:
                    await manager.send_message(session_id, {
                        "type": "frame_processed",
                        "success": False,
                        "message": result["error"],
                        "frames_collected": len(verification_frames),
                        "required_frames": face_service.required_frames
                    })
                    continue
                
                # Check quality
                if result["quality_score"] < face_service.min_quality_score:
                    await manager.send_message(session_id, {
                        "type": "frame_processed",
                        "success": False,
                        "message": f"Frame quality too low ({result['quality_score']:.1f}/100)",
                        "quality_score": result["quality_score"],
                        "frames_collected": len(verification_frames),
                        "required_frames": face_service.required_frames
                    })
                    continue
                
                # Compare with registered face
                comparison_result = face_service.compare_faces_with_verification(
                    registered_encoding,
                    result["encoding"]
                )
                
                # Store verification frame
                verification_frames.append({
                    "encoding": result["encoding"],
                    "quality_score": result["quality_score"],
                    "antispoofing_score": result["antispoofing_score"],
                    "similarity_score": comparison_result["similarity_score"],
                    "is_match": comparison_result["is_match"]
                })
                
                quality_scores.append(result["quality_score"])
                antispoofing_scores.append(result["antispoofing_score"])
                similarity_scores.append(comparison_result["similarity_score"])
                
                await manager.send_message(session_id, {
                    "type": "frame_processed",
                    "success": True,
                    "quality_score": result["quality_score"],
                    "antispoofing_score": result["antispoofing_score"],
                    "similarity_score": comparison_result["similarity_score"],
                    "is_match": comparison_result["is_match"],
                    "frames_collected": len(verification_frames),
                    "required_frames": face_service.required_frames,
                    "message": f"Frame verified! ({len(verification_frames)}/{face_service.required_frames})"
                })
                
                # Check if we have enough frames
                if len(verification_frames) >= face_service.required_frames:
                    # Calculate verification result
                    avg_quality = sum(quality_scores) / len(quality_scores)
                    avg_antispoofing = sum(antispoofing_scores) / len(antispoofing_scores)
                    avg_similarity = sum(similarity_scores) / len(similarity_scores)
                    
                    # Count matching frames
                    matching_frames = sum(1 for frame in verification_frames if frame["is_match"])
                    match_ratio = matching_frames / len(verification_frames)
                    
                    # Final verification decision
                    is_verified = (
                        avg_similarity >= 70.0 and 
                        match_ratio >= 0.7 and 
                        avg_antispoofing >= face_service.liveness_threshold
                    )
                    
                    # Log verification attempt
                    verification = FaceVerification(
                        UserID=user_id,
                        QuizID=quiz_id,
                        CourseID=course_id,
                        VerificationResult=is_verified,
                        SimilarityScore=avg_similarity,
                        Distance=1 - (avg_similarity / 100),
                        ThresholdUsed=70.0,
                        ModelName=face_service.model_name,
                        DistanceMetric=face_service.distance_metric,
                        ProcessingTime=sum(result.get("processing_time", 0) for result in verification_frames),
                        QualityScore=avg_quality
                    )
                    
                    db.add(verification)
                    
                    # Update streaming session
                    streaming_session.Status = "completed"
                    streaming_session.QualityScore = avg_quality
                    streaming_session.AntiSpoofingScore = avg_antispoofing
                    streaming_session.LivenessScore = match_ratio
                    streaming_session.EndTime = datetime.utcnow()
                    
                    # Update user's last login if verification successful
                    if is_verified:
                        user.LastLoginDateTime = datetime.utcnow()
                    
                    db.commit()
                    db.refresh(verification)
                    
                    await manager.send_message(session_id, {
                        "type": "verification_complete",
                        "success": True,
                        "verification_id": verification.id,
                        "user_name": user.Name,
                        "verified": is_verified,
                        "similarity_score": avg_similarity,
                        "quality_score": avg_quality,
                        "antispoofing_score": avg_antispoofing,
                        "match_ratio": match_ratio,
                        "frames_processed": len(verification_frames),
                        "message": "Identity verified successfully!" if is_verified else "Identity verification failed"
                    })
                    
                    if is_verified:
                        logger.info(f"✅ Face verification PASSED for user {user_id} via streaming")
                    else:
                        logger.warning(f"❌ Face verification FAILED for user {user_id} via streaming")
                    
                    break
                
                # Stop if we've processed too many frames
                if len(verification_frames) >= face_service.max_frames:
                    break
            
            elif message_data.get("type") == "stop":
                break
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Error in face verification stream: {str(e)}")
        try:
            await manager.send_message(session_id, {
                "type": "error",
                "message": f"Verification failed: {str(e)}"
            })
        except:
            pass
    finally:
        manager.disconnect(session_id)

# Keep existing API endpoints for backward compatibility and status checking
@app.get("/api/v1/face/status/{user_id}")
async def get_face_status(user_id: int, db: Session = Depends(get_db)):
    """Get face registration status for a user"""
    try:
        user = db.query(AppUser).filter(AppUser.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
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
        
        user = db.query(AppUser).filter(AppUser.id == user_id).first()
        user_name = user.Name if user else "Unknown"
        
        verification_list = []
        for v in verifications:
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
        db.execute(text("SELECT 1"))
        db_status = "operational"
    except:
        db_status = "error"
    
    active_sessions = len(manager.active_connections)
    
    return {
        "status": "healthy" if db_status == "operational" and s3_status in ["healthy", "not_configured"] else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "engine": "DeepFace",
        "streaming": "enabled",
        "anti_spoofing": face_service.anti_spoofing,
        "services": {
            "deepface": "operational",
            "database": db_status,
            "websocket": "operational",
            "streaming": "operational"
        },
        "configuration": {
            "model": face_service.model_name,
            "detector": face_service.detector_backend,
            "distance_metric": face_service.distance_metric,
            "anti_spoofing": face_service.anti_spoofing,
            "min_quality_score": face_service.min_quality_score,
            "liveness_threshold": face_service.liveness_threshold
        },
        "active_sessions": active_sessions,
        "version": "3.0.0"
    }

@app.get("/api/v1/stats")
async def get_system_stats(db: Session = Depends(get_db)):
    """Get system statistics including streaming data"""
    try:
        total_users = db.query(AppUser).filter(AppUser.Active == True).count()
        registered_faces = db.query(Face).filter(Face.IsActive == True).count()
        
        # Streaming sessions statistics
        from datetime import timedelta
        yesterday = datetime.utcnow() - timedelta(days=1)
        
        recent_sessions = db.query(StreamingSession).filter(
            StreamingSession.StartTime >= yesterday
        ).count()
        
        successful_registrations = db.query(StreamingSession).filter(
            StreamingSession.StartTime >= yesterday,
            StreamingSession.SessionType == "registration",
            StreamingSession.Status == "completed"
        ).count()
        
        successful_verifications = db.query(StreamingSession).filter(
            StreamingSession.StartTime >= yesterday,
            StreamingSession.SessionType == "verification",
            StreamingSession.Status == "completed"
        ).count()
        
        # Enhanced partition statistics
        partition_stats = {}
        if s3_service:
            users_per_partition = s3_service.users_per_partition
            max_partition = ((total_users - 1) // users_per_partition) + 1 if total_users > 0 else 0
            
            partition_stats = {
                "users_per_partition": users_per_partition,
                "active_partitions": max_partition,
                "partition_efficiency": (total_users / (max_partition * users_per_partition) * 100) if max_partition > 0 else 0,
                "max_supported_users": max_partition * users_per_partition,
                "next_partition_threshold": max_partition * users_per_partition + 1,
                "folder_structure": "partitioned" if s3_service else "flat"
            }
        
        return {
            "total_users": total_users,
            "registered_faces": registered_faces,
            "registration_rate": (registered_faces / total_users * 100) if total_users > 0 else 0,
            "active_streaming_sessions": len(manager.active_connections),
            "recent_sessions_24h": recent_sessions,
            "successful_registrations_24h": successful_registrations,
            "successful_verifications_24h": successful_verifications,
            "streaming_success_rate": (
                (successful_registrations + successful_verifications) / recent_sessions * 100
                if recent_sessions > 0 else 0
            ),
            "system_health": "excellent"
        }
        
    except Exception as e:
        logger.error(f"Error getting enhanced system stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get system statistics")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        workers=1,  # Single worker for WebSocket support
        log_level="info"
    )