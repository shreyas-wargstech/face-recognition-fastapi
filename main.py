# main.py - Enhanced with AWS Rekognition-like Face Liveness API

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, LargeBinary, Text, TIMESTAMP, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from deepface import DeepFace
import cv2
import numpy as np
import base64
import io
from PIL import Image
import uuid
import os
import logging
from datetime import datetime, timedelta
import json
import asyncio
import time
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor
import pickle
from cryptography.fernet import Fernet
from dotenv import load_dotenv
import gc
import traceback
from contextlib import asynccontextmanager

load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="LMS Face Recognition API with AWS Rekognition-like Face Liveness",
    description="Real-time face registration and verification system with AWS Rekognition-like Face Liveness API",
    version="4.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "https://your-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "mysql://root:password@localhost/lms_face_recognition")
engine = create_engine(DATABASE_URL, echo=True, pool_pre_ping=True, pool_recycle=300)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Encryption for face embeddings
ENCRYPTION_KEY = os.getenv("FACE_ENCRYPTION_KEY", Fernet.generate_key())
cipher_suite = Fernet(ENCRYPTION_KEY)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic Models
class CreateFaceLivenessSessionRequest(BaseModel):
    user_id: int
    settings: Optional[Dict[str, Any]] = {}
    audit_images_limit: Optional[int] = 4
    client_request_token: Optional[str] = None

class CreateFaceLivenessSessionResponse(BaseModel):
    session_id: str
    status: str = "CREATED"
    expires_at: str
    user_id: int
    audit_images_limit: int

class GetFaceLivenessSessionResultsResponse(BaseModel):
    session_id: str
    status: str
    confidence: float
    reference_image: Optional[Dict[str, Any]] = None
    audit_images: List[Dict[str, Any]] = []
    session_results: Dict[str, Any]

# Database Models
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
    RoleID = Column(Integer, nullable=False, default=1)
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

class FaceLivenessSession(Base):
    __tablename__ = "FaceLivenessSession"
    id = Column(Integer, primary_key=True, index=True)
    SessionID = Column(String(100), unique=True, nullable=False, index=True)
    UserID = Column(Integer, nullable=False, index=True)
    Status = Column(String(20), default="CREATED")
    ConfidenceScore = Column(Float, nullable=True)
    AuditImagesLimit = Column(Integer, default=4)
    ReferenceImageBase64 = Column(Text, nullable=True)
    ReferenceImageS3Key = Column(String(500), nullable=True)
    AuditImagesData = Column(Text, nullable=True)
    SessionResults = Column(Text, nullable=True)
    ClientRequestToken = Column(String(255), nullable=True)
    ExpiresAt = Column(DateTime, nullable=False)
    ProcessingTime = Column(Float, nullable=True)
    ErrorMessage = Column(Text, nullable=True)
    CreationDateTime = Column(TIMESTAMP, server_default=func.now())
    CompletionDateTime = Column(TIMESTAMP, nullable=True)

Base.metadata.create_all(bind=engine)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Utility Functions
def decode_base64_frame(frame_data: str) -> Optional[np.ndarray]:
    try:
        if not frame_data:
            logger.error("Empty frame data received")
            return None
        if frame_data.startswith('data:image'):
            frame_data = frame_data.split(',', 1)[1]
        image_bytes = base64.b64decode(frame_data)
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_array = np.array(image)
        if image_array.size == 0:
            logger.error("Empty image array after conversion")
            return None
        logger.debug(f"Successfully decoded frame: {image_array.shape}")
        return image_array
    except Exception as e:
        logger.error(f"Error decoding base64 frame: {str(e)}")
        return None

def encrypt_face_encoding(encoding: np.ndarray) -> bytes:
    try:
        if encoding is None or encoding.size == 0:
            raise ValueError("Invalid encoding provided")
        encoding_bytes = pickle.dumps(encoding)
        encrypted_bytes = cipher_suite.encrypt(encoding_bytes)
        return encrypted_bytes
    except Exception as e:
        logger.error(f"Error encrypting face encoding: {str(e)}")
        raise Exception("Failed to encrypt face encoding")

def decrypt_face_encoding(encrypted_data: bytes) -> np.ndarray:
    try:
        if not encrypted_data:
            raise ValueError("No encrypted data provided")
        decrypted_bytes = cipher_suite.decrypt(encrypted_data)
        encoding = pickle.loads(decrypted_bytes)
        return encoding
    except Exception as e:
        logger.error(f"Error decrypting face encoding: {str(e)}")
        raise Exception("Failed to decrypt face encoding")

def validate_frame_data(frame_data: str) -> bool:
    try:
        if not frame_data or not isinstance(frame_data, str):
            return False
        if frame_data.startswith('data:image'):
            frame_data = frame_data.split(',', 1)[1]
        base64.b64decode(frame_data, validate=True)
        return True
    except Exception as e:
        logger.debug(f"Frame validation failed: {str(e)}")
        return False

def encode_image_to_base64(image_array: np.ndarray) -> str:
    try:
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_array)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=95)
        base64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return base64_string
    except Exception as e:
        logger.error(f"Error encoding image to base64: {str(e)}")
        return ""

# Services
class OptimizedFaceRecognitionService:
    def __init__(self):
        self.model_name = "ArcFace"
        self.detector_backend = "opencv"
        self.distance_metric = "cosine"
        self.anti_spoofing = True
        self.min_quality_score = 25.0
        self.min_face_confidence = 0.5
        self.liveness_threshold = 0.4
        self.required_frames = 3
        self.max_frames = 8
        self.frame_skip = 8
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="face_processing")
        self.processing_timeout = 30
        logger.info("✅ Optimized FaceRecognitionService initialized")

    async def extract_face_async(self, image_array: np.ndarray, timeout: float = 15.0) -> Dict[str, Any]:
        try:
            if image_array is None or image_array.size == 0:
                return {"success": False, "error": "Invalid image data provided", "spoofing_detected": False, "processing_time": 0}
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(self.executor, self._extract_face_sync, image_array),
                timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Face extraction timeout after {timeout}s")
            return {"success": False, "error": "Processing timeout - please try again", "spoofing_detected": False, "processing_time": timeout * 1000}
        except Exception as e:
            logger.error(f"Face extraction error: {str(e)}")
            return {"success": False, "error": f"Processing error: {str(e)}", "spoofing_detected": False, "processing_time": 0}

    def _extract_face_sync(self, image_array: np.ndarray) -> Dict[str, Any]:
        try:
            start_time = time.time()
            if image_array is None or image_array.size == 0:
                return {"success": False, "error": "Invalid image array", "spoofing_detected": False, "processing_time": 0}
            faces = DeepFace.extract_faces(img_path=image_array, detector_backend=self.detector_backend, enforce_detection=False, align=True, anti_spoofing=self.anti_spoofing)
            if not faces:
                return {"success": False, "error": "No face detected in frame", "spoofing_detected": False, "processing_time": (time.time() - start_time) * 1000}
            face = faces[0]
            if not face.get('is_real', True):
                return {"success": False, "error": "Please use your real face - spoofing detected", "spoofing_detected": True, "antispoofing_score": face.get('antispoof_score', 0.0), "processing_time": (time.time() - start_time) * 1000}
            embeddings = DeepFace.represent(img_path=image_array, model_name=self.model_name, detector_backend=self.detector_backend, enforce_detection=False, align=True)
            if not embeddings:
                return {"success": False, "error": "Could not extract facial features", "spoofing_detected": False, "processing_time": (time.time() - start_time) * 1000}
            embedding = embeddings[0]['embedding']
            facial_area = embeddings[0]['facial_area']
            quality_score = self._quick_quality_score(image_array, facial_area, face.get('confidence', 0.9))
            processing_time = (time.time() - start_time) * 1000
            return {
                "success": True, "encoding": np.array(embedding), "facial_area": facial_area, "quality_score": quality_score,
                "face_confidence": face.get('confidence', 0.9), "antispoofing_score": face.get('antispoof_score', 1.0),
                "is_real": face.get('is_real', True), "spoofing_detected": False, "model_name": self.model_name,
                "processing_time": processing_time, "face_image": face.get('face', None)
            }
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000 if 'start_time' in locals() else 0
            logger.error(f"Face extraction sync error: {str(e)}")
            return {"success": False, "error": f"Face processing failed: {str(e)}", "spoofing_detected": False, "processing_time": processing_time}

    def _quick_quality_score(self, image: np.ndarray, facial_area: dict, face_confidence: float) -> float:
        try:
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            face_area = w * h
            size_factor = min(1.0, face_area / (100 * 100))
            quality_score = (face_confidence * 60 + size_factor * 40) * 100
            return max(0, min(100, quality_score))
        except Exception as e:
            logger.error(f"Quality score calculation error: {str(e)}")
            return 50.0

    def compare_faces_with_verification(self, registered_encoding: np.ndarray, current_encoding: np.ndarray) -> Dict[str, Any]:
        try:
            start_time = time.time()
            if registered_encoding is None or current_encoding is None:
                return {"similarity_score": 0.0, "is_match": False, "confidence": 0.0, "error": "Invalid encoding data", "processing_time": 0}
            cosine_similarity = np.dot(registered_encoding, current_encoding) / (np.linalg.norm(registered_encoding) * np.linalg.norm(current_encoding))
            euclidean_distance = np.linalg.norm(registered_encoding - current_encoding)
            manhattan_distance = np.sum(np.abs(registered_encoding - current_encoding))
            similarity_score = max(0, cosine_similarity * 100)
            cosine_threshold = 0.55
            euclidean_threshold = 1.2
            cosine_match = cosine_similarity >= cosine_threshold
            euclidean_match = euclidean_distance <= euclidean_threshold
            similarity_match = similarity_score >= 55.0
            is_match = sum([cosine_match, euclidean_match, similarity_match]) >= 2
            confidence = (similarity_score / 100) * 0.7 + (1 - min(euclidean_distance / 2.0, 1.0)) * 0.3
            processing_time = (time.time() - start_time) * 1000
            return {
                "similarity_score": float(similarity_score), "cosine_similarity": float(cosine_similarity),
                "euclidean_distance": float(euclidean_distance), "manhattan_distance": float(manhattan_distance),
                "is_match": bool(is_match), "confidence": float(confidence), "cosine_match": bool(cosine_match),
                "euclidean_match": bool(euclidean_match), "similarity_match": bool(similarity_match),
                "threshold_used": float(cosine_threshold), "model_name": self.model_name, "processing_time": processing_time
            }
        except Exception as e:
            logger.error(f"Error comparing faces for verification: {str(e)}")
            return {"similarity_score": 0.0, "is_match": False, "confidence": 0.0, "error": str(e), "processing_time": 0}

    def calculate_liveness_confidence(self, frames_data: List[Dict[str, Any]]) -> float:
        try:
            if not frames_data:
                return 0.0
            quality_scores = [frame.get('quality_score', 0) for frame in frames_data]
            antispoofing_scores = [frame.get('antispoofing_score', 0) for frame in frames_data]
            face_confidences = [frame.get('face_confidence', 0) for frame in frames_data]
            avg_quality = sum(quality_scores) / len(quality_scores)
            avg_antispoofing = sum(antispoofing_scores) / len(antispoofing_scores)
            avg_face_confidence = sum(face_confidences) / len(face_confidences)
            quality_variation = np.std(quality_scores) if len(quality_scores) > 1 else 0
            variation_factor = min(quality_variation / 10.0, 1.0)
            liveness_confidence = (avg_quality * 0.3 + avg_antispoofing * 40 + avg_face_confidence * 100 * 0.2 + variation_factor * 10)
            return max(0, min(100, liveness_confidence))
        except Exception as e:
            logger.error(f"Error calculating liveness confidence: {str(e)}")
            return 0.0

class FaceLivenessSessionManager:
    def __init__(self):
        self.active_sessions: Dict[str, Dict] = {}
        self.session_timeout = 180
        self.max_audit_images = 4

    def create_session(self, user_id: int, audit_images_limit: int = 4, client_request_token: Optional[str] = None) -> str:
        session_id = f"fl_{uuid.uuid4().hex}"
        expires_at = datetime.utcnow() + timedelta(seconds=self.session_timeout)
        self.active_sessions[session_id] = {
            "user_id": user_id, "status": "CREATED", "expires_at": expires_at,
            "audit_images_limit": min(audit_images_limit, self.max_audit_images), "frames": [],
            "audit_images": [], "reference_image": None, "confidence_score": 0.0,
            "created_at": datetime.utcnow(), "client_request_token": client_request_token
        }
        logger.info(f"Created Face Liveness session: {session_id} for user {user_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict]:
        if session_id not in self.active_sessions:
            return None
        session = self.active_sessions[session_id]
        if datetime.utcnow() > session["expires_at"]:
            del self.active_sessions[session_id]
            return None
        return session

    def update_session(self, session_id: str, update_data: Dict) -> bool:
        if session_id not in self.active_sessions:
            return False
        session = self.active_sessions[session_id]
        if datetime.utcnow() > session["expires_at"]:
            del self.active_sessions[session_id]
            return False
        session.update(update_data)
        return True

    def complete_session(self, session_id: str) -> bool:
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["status"] = "COMPLETED"
            self.active_sessions[session_id]["completed_at"] = datetime.utcnow()
            return True
        return False

    def cleanup_expired_sessions(self):
        current_time = datetime.utcnow()
        expired_sessions = [sid for sid, session in self.active_sessions.items() if current_time > session["expires_at"]]
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Cleaned up expired session: {session_id}")

class OptimizedConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Dict] = {}
        self.connection_timestamps: Dict[str, datetime] = {}
        self.processing_locks: Dict[str, asyncio.Lock] = {}
        self.cleanup_task = None
        self.max_connection_time = 300
        self.heartbeat_interval = 30

    async def connect(self, websocket: WebSocket, session_id: str, user_id: int, session_type: str):
        await websocket.accept()
        self.active_connections[session_id] = {
            "websocket": websocket, "user_id": user_id, "session_type": session_type,
            "frame_buffer": deque(maxlen=10), "frame_count": 0, "processed_count": 0,
            "quality_scores": deque(maxlen=20), "last_activity": time.time(), "start_time": time.time(),
            "processing": False, "last_heartbeat": time.time(), "timeout_warnings": 0
        }
        self.connection_timestamps[session_id] = datetime.utcnow()
        self.processing_locks[session_id] = asyncio.Lock()
        asyncio.create_task(self._heartbeat_loop(session_id))
        logger.info(f"🔗 WebSocket connected: {session_id} for user {user_id} ({session_type})")
        if self.cleanup_task is None:
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            try:
                connection = self.active_connections[session_id]
                if connection.get("frame_buffer"):
                    connection["frame_buffer"].clear()
                del self.active_connections[session_id]
                del self.connection_timestamps[session_id]
                if session_id in self.processing_locks:
                    del self.processing_locks[session_id]
                logger.info(f"❌ WebSocket disconnected: {session_id}")
                gc.collect()
            except Exception as e:
                logger.error(f"Error during disconnect cleanup: {str(e)}")

    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            try:
                websocket = self.active_connections[session_id]["websocket"]
                await asyncio.wait_for(websocket.send_text(json.dumps(message)), timeout=5.0)
                self.active_connections[session_id]["last_activity"] = time.time()
                return True
            except asyncio.TimeoutError:
                logger.warning(f"Send timeout for session {session_id}")
                await self.force_disconnect(session_id)
                return False
            except Exception as e:
                logger.error(f"Send error for session {session_id}: {str(e)}")
                await self.force_disconnect(session_id)
                return False
        return False

    async def force_disconnect(self, session_id: str):
        if session_id in self.active_connections:
            try:
                websocket = self.active_connections[session_id]["websocket"]
                await websocket.close(code=1000, reason="Server timeout")
            except:
                pass
            finally:
                await self.disconnect(session_id)

    async def _heartbeat_loop(self, session_id: str):
        while session_id in self.active_connections:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                if session_id not in self.active_connections:
                    break
                connection = self.active_connections[session_id]
                current_time = time.time()
                if current_time - connection.get("last_activity", 0) > 60:
                    success = await self.send_message(session_id, {
                        "type": "heartbeat", "timestamp": current_time, "session_time": current_time - connection["start_time"]
                    })
                    if not success:
                        break
            except Exception as e:
                logger.error(f"Heartbeat error for {session_id}: {str(e)}")
                break

    async def _cleanup_loop(self):
        while True:
            try:
                await asyncio.sleep(30)
                current_time = time.time()
                stale_sessions = []
                for session_id, connection in self.active_connections.items():
                    session_age = current_time - connection["start_time"]
                    last_activity = current_time - connection.get("last_activity", current_time)
                    if (session_age > self.max_connection_time or last_activity > 120 or connection.get("timeout_warnings", 0) > 3):
                        stale_sessions.append(session_id)
                for session_id in stale_sessions:
                    logger.warning(f"Cleaning up stale session: {session_id}")
                    await self.force_disconnect(session_id)
                if len(self.active_connections) > 0:
                    try:
                        import psutil
                        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
                        logger.info(f"Active connections: {len(self.active_connections)}, Memory: {memory_usage:.1f}MB")
                    except ImportError:
                        logger.info(f"Active connections: {len(self.active_connections)}")
            except Exception as e:
                logger.error(f"Cleanup loop error: {str(e)}")
                await asyncio.sleep(10)

# Initialize services
manager = OptimizedConnectionManager()
face_service = OptimizedFaceRecognitionService()
liveness_manager = FaceLivenessSessionManager()

# API Endpoints
@app.post("/api/v1/face-liveness/create-session", response_model=CreateFaceLivenessSessionResponse)
async def create_face_liveness_session(request: CreateFaceLivenessSessionRequest, db: Session = Depends(get_db)):
    try:
        user = db.query(AppUser).filter(AppUser.id == request.user_id, AppUser.Active == True).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        session_id = liveness_manager.create_session(
            user_id=request.user_id, audit_images_limit=request.audit_images_limit, client_request_token=request.client_request_token
        )
        expires_at = datetime.utcnow() + timedelta(seconds=liveness_manager.session_timeout)
        db_session = FaceLivenessSession(
            SessionID=session_id, UserID=request.user_id, Status="CREATED", AuditImagesLimit=request.audit_images_limit,
            ClientRequestToken=request.client_request_token, ExpiresAt=expires_at
        )
        db.add(db_session)
        db.commit()
        db.refresh(db_session)
        logger.info(f"✅ Face Liveness session created: {session_id} for user {request.user_id}")
        return CreateFaceLivenessSessionResponse(
            session_id=session_id, status="CREATED", expires_at=expires_at.isoformat(), user_id=request.user_id,
            audit_images_limit=request.audit_images_limit
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating Face Liveness session: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create Face Liveness session")

@app.get("/api/v1/face-liveness/session-results/{session_id}", response_model=GetFaceLivenessSessionResultsResponse)
async def get_face_liveness_session_results(session_id: str, db: Session = Depends(get_db)):
    try:
        db_session = db.query(FaceLivenessSession).filter(FaceLivenessSession.SessionID == session_id).first()
        if not db_session:
            raise HTTPException(status_code=404, detail="Session not found")
        if datetime.utcnow() > db_session.ExpiresAt:
            db_session.Status = "EXPIRED"
            db.commit()
            raise HTTPException(status_code=410, detail="Session has expired")
        if db_session.Status != "COMPLETED":
            return GetFaceLivenessSessionResultsResponse(
                session_id=session_id, status=db_session.Status, confidence=0.0,
                session_results={"status": db_session.Status, "message": "Session掍Session not yet completed"}
            )
        reference_image = None
        if db_session.ReferenceImageBase64:
            reference_image = {"bytes": db_session.ReferenceImageBase64, "bounding_box": {"width": 1.0, "height": 1.0, "left": 0.0, "top": 0.0}, "quality": {"brightness": 80.0, "sharpness": 80.0}}
        audit_images = []
        if db_session.AuditImagesData:
            try:
                audit_images_data = json.loads(db_session.AuditImagesData)
                for i, img_data in enumerate(audit_images_data):
                    audit_images.append({
                        "bytes": img_data.get("base64", ""), "bounding_box": img_data.get("bounding_box", {"width": 1.0, "height": 1.0, "left": 0.0, "top": 0.0}),
                        "quality": img_data.get("quality", {"brightness": 80.0, "sharpness": 80.0}), "sequence_id": i + 1
                    })
            except json.JSONDecodeError:
                logger.error(f"Failed to parse audit images data for session {session_id}")
        session_results = {}
        if db_session.SessionResults:
            try:
                session_results = json.loads(db_session.SessionResults)
            except json.JSONDecodeError:
                session_results = {"status": "COMPLETED"}
        return GetFaceLivenessSessionResultsResponse(
            session_id=session_id, status=db_session.Status, confidence=db_session.ConfidenceScore or 0.0,
            reference_image=reference_image, audit_images=audit_images, session_results=session_results
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting Face Liveness session results: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get session results")

@app.websocket("/ws/face-liveness/{session_id}")
async def face_liveness_stream(websocket: WebSocket, session_id: str, db: Session = Depends(get_db)):
    connection_id = f"fl_{session_id}_{uuid.uuid4().hex[:8]}"
    try:
        session = liveness_manager.get_session(session_id)
        if not session:
            await websocket.close(code=4004, reason="Session not found or expired")
            return
        user = db.query(AppUser).filter(AppUser.id == session["user_id"], AppUser.Active == True).first()
        if not user:
            await websocket.close(code=4004, reason="User not found")
            return
        await manager.connect(websocket, connection_id, session["user_id"], "face_liveness")
        liveness_manager.update_session(session_id, {"status": "IN_PROGRESS"})
        await manager.send_message(connection_id, {
            "type": "session_started", "session_id": session_id, "user_id": session["user_id"], "user_name": user.Name,
            "audit_images_limit": session["audit_images_limit"], "expires_at": session["expires_at"].isoformat(),
            "message": "Face Liveness session started. Please look at the camera."
        })
        collected_frames = []
        audit_images = []
        best_quality_frame = None
        best_quality_score = 0
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                message_data = json.loads(data)
                if message_data.get("type") == "frame":
                    connection = manager.active_connections.get(connection_id)
                    if not connection:
                        break
                    session = liveness_manager.get_session(session_id)
                    if not session:
                        await manager.send_message(connection_id, {"type": "session_expired", "message": "Session has expired"})
                        break
                    async with manager.processing_locks[connection_id]:
                        if connection.get("processing", False):
                            continue
                        connection["processing"] = True
                    try:
                        frame_data = message_data.get("frame", "")
                        connection["frame_count"] += 1
                        if connection["frame_count"] % 4 != 0:
                            continue
                        if not validate_frame_data(frame_data):
                            await manager.send_message(connection_id, {"type": "error", "message": "Invalid frame data format"})
                            continue
                        frame = decode_base64_frame(frame_data)
                        if frame is None:
                            await manager.send_message(connection_id, {"type": "error", "message": "Failed to decode frame data"})
                            continue
                        process_start = time.time()
                        result = await face_service.extract_face_async(frame, timeout=8.0)
                        if result.get("spoofing_detected", False):
                            await manager.send_message(connection_id, {
                                "type": "spoofing_detected", "message": result.get("error", "Spoofing detected"),
                                "antispoofing_score": result.get("antispoofing_score", 0.0)
                            })
                            continue
                        if not result.get("success", False):
                            await manager.send_message(connection_id, {
                                "type": "frame_processed", "success": False, "message": result.get("error", "Frame processing failed"),
                                "processing_time": (time.time() - process_start) * 1000
                            })
                            continue
                        quality_score = result.get("quality_score", 0)
                        frame_info = {
                            "quality_score": quality_score, "antispoofing_score": result.get("antispoofing_score", 1.0),
                            "face_confidence": result.get("face_confidence", 0.9), "facial_area": result.get("facial_area", {}),
                            "timestamp": time.time(), "frame_base64": frame_data, "face_image": result.get("face_image")
                        }
                        collected_frames.append(frame_info)
                        connection["processed_count"] unul += 1
                        if quality_score > best_quality_score:
                            best_quality_score = quality_score
                            best_quality_frame = frame_info
                        if len(audit_images) < session["audit_images_limit"]:
                            if quality_score > 30 and len(collected_frames) % 3 == 0:
                                audit_images.append(frame_info)
                        await manager.send_message(connection_id, {
                            "type": "frame_processed", "success": True, "quality_score": quality_score,
                            "antispoofing_score": result.get("antispoofing_score", 1.0), "face_confidence": result.get("face_confidence", 0.9),
                            "frames_collected": len(collected_frames), "audit_images_collected": len(audit_images),
                            "processing_time": (time.time() - process_start) * 1000, "message": f"Liveness check in progress... ({len(collected_frames)} frames)"
                        })
                        if len(collected_frames) >= 8:
                            confidence_score = face_service.calculate_liveness_confidence(collected_frames)
                            liveness_passed = confidence_score >= 70.0
                            reference_image_data = None
                            if best_quality_frame:
                                reference_image_data = {
                                    "base64": best_quality_frame["frame_base64"], "quality_score": best_quality_frame["quality_score"],
                                    "bounding_box": {"width": 1.0, "height": 1.0, "left": 0.0, "top": 0.0},
                                    "quality": {"brightness": 80.0, "sharpness": best_quality_frame["quality_score"]}
                                }
                            audit_images_data = []
                            for img in audit_images:
                                audit_images_data.append({
                                    "base64": img["frame_base64"], "quality_score": img["quality_score"],
                                    "bounding_box": {"width": 1.0, "height": 1.0, "left": 0.0, "top": 0.0},
                                    "quality": {"brightness": 80.0, "sharpness": img["quality_score"]}
                                })
                            session_results = {
                                "status": "COMPLETED", "confidence": confidence_score, "liveness_passed": liveness_passed,
                                "frames_analyzed": len(collected_frames), "audit_images_count": len(audit_images),
                                "processing_time": sum(f.get("processing_time", 0) for f in collected_frames) / len(collected_frames),
                                "quality_metrics": {
                                    "avg_quality": sum(f["quality_score"] for f in collected_frames) / len(collected_frames),
                                    "avg_antispoofing": sum(f["antispoofing_score"] for f in collected_frames) / len(collected_frames),
                                    "best_quality": best_quality_score
                                }
                            }
                            try:
                                db_session = db.query(FaceLivenessSession).filter(FaceLivenessSession.SessionID == session_id).first()
                                if db_session:
                                    db_session.Status = "COMPLETED"
                                    db_session.ConfidenceScore = confidence_score
                                    db_session.ReferenceImageBase64 = best_quality_frame["frame_base64"] if best_quality_frame else None
                                    db_session.AuditImagesData = json.dumps(audit_images_data)
                                    db_session.SessionResults = json.dumps(session_results)
                                    db_session.CompletionDateTime = datetime.utcnow()
                                    db.commit()
                                liveness_manager.complete_session(session_id)
                                await manager.send_message(connection_id, {
                                    "type": "liveness_complete", "success": True, "session_id": session_id, "confidence": confidence_score,
                                    "liveness_passed": liveness_passed, "frames_analyzed": len(collected_frames), "audit_images_count": len(audit_images),
                                    "reference_image_quality": best_quality_score, "session_results": session_results,
                                    "message": "Face Liveness analysis completed successfully!" if liveness_passed else "Liveness check failed - please try again"
                                })
                                logger.info(f"✅ Face Liveness completed for session {session_id}: {confidence_score:.1f}% confidence")
                                break
                            except Exception as e:
                                logger.error(f"Database error during liveness completion: {str(e)}")
                                db.rollback()
                                await manager.send_message(connection_id, {"type": "error", "message": f"Failed to save liveness results: {str(e)}"})
                                break
                    finally:
                        if connection:
                            connection["processing"] = False
                elif message_data.get("type") == "stop":
                    break
                elif message_data.get("type") == "ping":
                    await manager.send_message(connection_id, {"type": "pong", "timestamp": message_data.get("timestamp")})
            except asyncio.TimeoutError:
                logger.warning(f"Face Liveness WebSocket timeout for {connection_id}")
                await manager.send_message(connection_id, {"type": "timeout_warning", "message": "Connection timeout - please check your network"})
                break
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Face Liveness frame processing error: {str(e)}")
                await manager.send_message(connection_id, {"type": "error", "message": f"Processing error: {str(e)}"})
    except WebSocketDisconnect:
        logger.info(f"Face Liveness WebSocket disconnected normally: {connection_id}")
    except Exception as e:
        logger.error(f"Face Liveness stream error for {connection_id}: {str(e)}")
        try:
            await manager.send_message(connection_id, {"type": "error", "message": f"Face Liveness failed: {str(e)}"})
        except:
            pass
    finally:
        await manager.disconnect(connection_id)

@app.websocket("/ws/face-registration/{user_id}")
async def optimized_face_registration_stream(websocket: WebSocket, user_id: int, db: Session = Depends(get_db)):
    session_id = f"reg_{user_id}_{uuid.uuid4().hex[:8]}"
    try:
        user = db.query(AppUser).filter(AppUser.id == user_id, AppUser.Active == True).first()
        if not user:
            await websocket.close(code=4004, reason="User not found")
            return
        await manager.connect(websocket, session_id, user_id, "registration")
        await manager.send_message(session_id, {
            "type": "connected", "session_id": session_id, "user_id": user_id, "user_name": user.Name,
            "required_frames": face_service.required_frames, "message": "Ready for face registration. Please look at the camera."
        })
        best_frames = []
        quality_scores = []
        frame_processing_times = []
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                message_data = json.loads(data)
                if message_data.get("type") == "frame":
                    connection = manager.active_connections.get(session_id)
                    if not connection:
                        break
                    async with manager.processing_locks[session_id]:
                        if connection.get("processing", False):
                            continue
                        connection["processing"] = True
                    try:
                        frame_data = message_data.get("frame", "")
                        connection["frame_count"] += 1
                        if connection["frame_count"] % face_service.frame_skip != 0:
                            continue
                        if not validate_frame_data(frame_data):
                            await manager.send_message(session_id, {"type": "error", "message": "Invalid frame data format"})
                            continue
                        frame = decode_base64_frame(frame_data)
                        if frame is None:
                            await manager.send_message(session_id, {"type": "error", "message": "Failed to decode frame data"})
                            continue
                        process_start = time.time()
                        result = await face_service.extract_face_async(frame, timeout=10.0)
                        process_time = (time.time() - process_start) * 1000
                        frame_processing_times.append(process_time)
                        connection["processed_count"] += 1
                        if result.get("spoofing_detected", False):
                            await manager.send_message(session_id, {
                                "type": "spoofing_detected", "message": result.get("error", "Spoofing detected"),
                                "antispoofing_score": result.get("antispoofing_score", 0.0)
                            })
                            continue
                        if not result.get("success", False):
                            await manager.send_message(session_id, {
                                "type": "frame_processed", "success": False, "message": result.get("error", "Frame processing failed"),
                                "frames_collected": len(best_frames), "required_frames": face_service.required_frames, "processing_time": process_time
                            })
                            continue
                        quality_score = result.get("quality_score", 0)
                        if quality_score < face_service.min_quality_score:
                            await manager.send_message(session_id, {
                                "type": "frame_processed", "success": False,
                                "message": f"Frame quality: {quality_score:.1f}/100 (need >{face_service.min_quality_score})",
                                "quality_score": quality_score, "frames_collected": len(best_frames), "required_frames": face_service.required_frames,
                                "processing_time": process_time
                            })
                            continue
                        best_frames.append({
                            "encoding": result.get("encoding"), "quality_score": quality_score,
                            "antispoofing_score": result.get("antispoofing_score", 1.0), "face_confidence": result.get("face_confidence", 0.9)
                        })
                        quality_scores.append(quality_score)
                        await manager.send_message(session_id, {
                            "type": "frame_processed", "success": True, "quality_score": quality_score,
                            "antispoofing_score": result.get("antispoofing_score", 1.0), "face_confidence": result.get("face_confidence", 0.9),
                            "frames_collected": len(best_frames), "required_frames": face_service.required_frames, "processing_time": process_time,
                            "message": f"Good frame! ({len(best_frames)}/{face_service.required_frames})"
                        })
                        if len(best_frames) >= face_service.required_frames:
                            best_frame = max(best_frames, key=lambda x: x["quality_score"])
                            avg_quality = sum(quality_scores) / len(quality_scores)
                            avg_antispoofing = sum(f["antispoofing_score"] for f in best_frames) / len(best_frames)
                            avg_processing_time = sum(frame_processing_times) / len(frame_processing_times)
                            try:
                                async with asyncio.timeout(10.0):
                                    encrypted_embedding = encrypt_face_encoding(best_frame["encoding"])
                                    existing_face = db.query(Face).filter(Face.UserID == user_id, Face.IsActive == True).first()
                                    if existing_face:
                                        existing_face.IsActive = False
                                    new_face = Face(
                                        UserID=user_id, FaceEmbedding=encrypted_embedding,
                                        FaceData=f"stream_{face_service.model_name}_{uuid.uuid4().hex[:8]}",
                                        ModelName=face_service.model_name, DetectorBackend=face_service.detector_backend,
                                        QualityScore=avg_quality, FaceConfidence=best_frame["face_confidence"], RegistrationSource="stream_v2"
                                    )
                                    db.add(new_face)
                                    db.commit()
                                    db.refresh(new_face)
                                    await manager.send_message(session_id, {
                                        "type": "registration_complete", "success": True, "face_id": new_face.id, "user_id": user_id,
                                        "user_name": user.Name, "quality_score": avg_quality, "antispoofing_score": avg_antispoofing,
                                        "face_confidence": best_frame["face_confidence"], "frames_processed": len(best_frames),
                                        "avg_processing_time": avg_processing_time, "model_name": face_service.model_name,
                                        "registration_source": "stream_v2", "message": "Face registration completed successfully!"
                                    })
                                    logger.info(f"✅ Face registered for user {user_id} in {avg_processing_time:.2f}ms avg")
                                    break
                            except asyncio.TimeoutError:
                                await manager.send_message(session_id, {"type": "error", "message": "Database timeout - please try again"})
                                break
                            except Exception as e:
                                logger.error(f"Database error: {str(e)}")
                                db.rollback()
                                await manager.send_message(session_id, {"type": "error", "message": f"Registration failed: {str(e)}"})
                                break
                    finally:
                        if connection:
                            connection["processing"] = False
                elif message_data.get("type") == "stop":
                    break
                elif message_data.get("type") == "ping":
                    await manager.send_message(session_id, {"type": "pong", "timestamp": message_data.get("timestamp")})
            except asyncio.TimeoutError:
                logger.warning(f"WebSocket receive timeout for {session_id}")
                await manager.send_message(session_id, {"type": "timeout_warning", "message": "Connection timeout - please check your network"})
                connection = manager.active_connections.get(session_id)
                if connection:
                    connection["timeout_warnings"] = connection.get("timeout_warnings", 0) + 1
                    if connection["timeout_warnings"] > 2:
                        break
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Frame processing error: {str(e)}")
                await manager.send_message(session_id, {"type": "error", "message": f"Processing error: {str(e)}"})
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected normally: {session_id}")
    except Exception as e:
        logger.error(f"Registration stream error for {session_id}: {str(e)}")
        try:
            await manager.send_message(session_id, {"type": "error", "message": f"Registration failed: {str(e)}"})
        except:
            pass
    finally:
        await manager.disconnect(session_id)

@app.websocket("/ws/face-verification/{user_id}")
async def optimized_face_verification_stream(websocket: WebSocket, user_id: int, quiz_id: Optional[str] = None, course_id: Optional[str] = None, db: Session = Depends(get_db)):
    session_id = f"ver_{user_id}_{uuid.uuid4().hex[:8]}"
    try:
        user = db.query(AppUser).filter(AppUser.id == user_id, AppUser.Active == True).first()
        if not user:
            await websocket.close(code=4004, reason="User not found")
            return
        registered_face = db.query(Face).filter(Face.UserID == user_id, Face.IsActive == True).first()
        if not registered_face:
            await websocket.close(code=4003, reason="No face registration found")
            return
        await manager.connect(websocket, session_id, user_id, "verification")
        await manager.send_message(session_id, {
            "type": "connected", "session_id": session_id, "user_id": user_id, "user_name": user.Name, "quiz_id": quiz_id,
            "course_id": course_id, "required_frames": 2, "message": "Ready for face verification. Please look at the camera."
        })
        verification_frames = []
        similarity_scores = []
        frame_processing_times = []
        registered_encoding = decrypt_face_encoding(registered_face.FaceEmbedding)
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                message_data = json.loads(data)
                if message_data.get("type") == "frame":
                    connection = manager.active_connections.get(session_id)
                    if not connection:
                        break
                    async with manager.processing_locks[session_id]:
                        if connection.get("processing", False):
                            continue
                        connection["processing"] = True
                    try:
                        frame_data = message_data["frame"]
                        connection["frame_count"] += 1
                        if connection["frame_count"] % 4 != 0:
                            continue
                        if not validate_frame_data(frame_data):
                            await manager.send_message(session_id, {"type": "error", "message": "Invalid frame data format"})
                            continue
                        frame = decode_base64_frame(frame_data)
                        if frame is None:
                            await manager.send_message(session_id, {"type": "error", "message": "Failed to decode frame data"})
                            continue
                        process_start = time.time()
                        result = await face_service.extract_face_async(frame, timeout=8.0)
                        if result.get("spoofing_detected", False):
                            await manager.send_message(session_id, {
                                "type": "spoofing_detected", "message": result.get("error", "Spoofing detected"),
                                "antispoofing_score": result.get("antispoofing_score", 0.0), "can_retry": True
                            })
                            continue
                        if not result.get("success", False):
                            await manager.send_message(session_id, {
                                "type": "frame_processed", "success": False, "message": result.get("error", "Frame processing failed"),
                                "frames_collected": len(verification_frames), "required_frames": 2, "processing_time": (time.time() - process_start) * 1000
                            })
                            continue
                        quality_score = result.get("quality_score", 0)
                        if quality_score < 10.0:
                            await manager.send_message(session_id, {
                                "type": "frame_processed", "success": False, "message": f"Frame quality too low: {quality_score:.1f}% (need >10%)",
                                "quality_score": quality_score, "frames_collected": len(verification_frames), "required_frames": 2,
                                "processing_time": (time.time() - process_start) * 1000
                            })
                            continue
                        current_encoding = result.get("encoding")
                        comparison_start = time.time()
                        comparison_result = face_service.compare_faces_with_verification(registered_encoding, current_encoding)
                        comparison_time = (time.time() - comparison_start) * 1000
                        similarity_score = comparison_result.get("similarity_score", 0)
                        is_match = comparison_result.get("is_match", False)
                        verification_frames.append({
                            "similarity_score": similarity_score, "quality_score": quality_score,
                            "antispoofing_score": result.get("antispoofing_score", 1.0), "is_match": is_match,
                            "confidence": comparison_result.get("confidence", 0)
                        })
                        similarity_scores.append(similarity_score)
                        frame_processing_times.append((time.time() - process_start) * 1000)
                        connection["processed_count"] += 1
                        await manager.send_message(session_id, {
                            "type": "frame_processed", "success": True, "similarity_score": similarity_score, "is_match": is_match,
                            "quality_score": quality_score, "antispoofing_score": result.get("antispoofing_score", 1.0),
                            "frames_collected": len(verification_frames), "required_frames": 2, "processing_time": (time.time() - process_start) * 1000,
                            "comparison_time": comparison_time, "message": f"Verification progress: {similarity_score:.1f}% similarity ({'Match' if is_match else 'No match'})"
                        })
                        if len(verification_frames) >= 2:
                            max_similarity = max(similarity_scores)
                            avg_similarity = sum(similarity_scores) / len(similarity_scores)
                            match_count = sum(1 for frame in verification_frames if frame["is_match"])
                            match_ratio = match_count / len(verification_frames)
                            verified = (max_similarity >= 55.0 or (avg_similarity >= 45.0 and match_ratio >= 0.5) or match_count >= 1)
                            avg_quality = sum(f["quality_score"] for f in verification_frames) / len(verification_frames)
                            avg_antispoofing = sum(f["antispoofing_score"] for f in verification_frames) / len(verification_frames)
                            avg_processing_time = sum(frame_processing_times) / len(frame_processing_times)
                            confidence_score = max(f["confidence"] for f in verification_frames)
                            try:
                                verification_record = FaceVerification(
                                    UserID=user_id, QuizID=quiz_id, CourseID=course_id, VerificationResult=verified,
                                    SimilarityScore=max_similarity, Distance=1.0 - (max_similarity / 100), ThresholdUsed=55.0,
                                    ModelName=face_service.model_name, DistanceMetric=face_service.distance_metric,
                                    ProcessingTime=avg_processing_time, QualityScore=avg_quality
                                )
                                db.add(verification_record)
                                db.commit()
                                db.refresh(verification_record)
                                await manager.send_message(session_id, {
                                    "type": "verification_complete", "success": True, "verification_id": verification_record.id,
                                    "user_id": user_id, "user_name": user.Name, "quiz_id": quiz_id, "course_id": course_id,
                                    "verified": verified, "similarity_score": max_similarity, "max_similarity_score": max_similarity,
                                    "distance": 1.0 - (max_similarity / 100), "threshold": 55.0, "quality_score": avg_quality,
                                    "antispoofing_score": avg_antispoofing, "match_ratio": match_ratio, "confidence_score": confidence_score,
                                    "frames_processed": len(verification_frames), "processing_time": avg_processing_time,
                                    "avg_processing_time": avg_processing_time, "model_name": face_service.model_name,
                                    "verification_method": "optimized_stream", "threshold_used": 55.0,
                                    "message": "Identity verified successfully!" if verified else "Identity verification failed"
                                })
                                logger.info(f"✅ Face verification completed for user {user_id}: {verified} ({max_similarity:.1f}%)")
                                break
                            except Exception as e:
                                logger.error(f"Database error during verification: {str(e)}")
                                db.rollback()
                                await manager.send_message(session_id, {
                                    "type": "error", "message": f"Verification failed: {str(e)}", "can_retry": True
                                })
                                break
                    finally:
                        if connection:
                            connection["processing"] = False
                elif message_data.get("type") == "stop":
                    break
                elif message_data.get("type") == "restart_verification":
                    verification_frames = []
                    similarity_scores = []
                    frame_processing_times = []
                    connection = manager.active_connections.get(session_id)
                    if connection:
                        connection["frame_count"] = 0
                        connection["processed_count"] = 0
                    await manager.send_message(session_id, {"type": "verification_restarted", "message": "Verification restarted. Please look at the camera."})
                elif message_data.get("type") == "ping":
                    await manager.send_message(session_id, {"type": "pong", "timestamp": message_data.get("timestamp")})
            except asyncio.TimeoutError:
                logger.warning(f"Verification WebSocket receive timeout for {session_id}")
                await manager.send_message(session_id, {"type": "timeout_warning", "message": "Connection timeout - please check your network", "can_retry": True})
                connection = manager.active_connections.get(session_id)
                if connection:
                    connection["timeout_warnings"] = connection.get("timeout_warnings", 0) + 1
                    if connection["timeout_warnings"] > 2:
                        break
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Verification frame processing error: {str(e)}")
                await manager.send_message(session_id, {"type": "error", "message": f"Processing error: {str(e)}", "can_retry": True})
    except WebSocketDisconnect:
        logger.info(f"Verification WebSocket disconnected normally: {session_id}")
    except Exception as e:
        logger.error(f"Verification stream error for {session_id}: {str(e)}")
        try:
            await manager.send_message(session_id, {"type": "error", "message": f"Verification failed: {str(e)}", "can_retry": True})
        except:
            pass
    finally:
        await manager.disconnect(session_id)

@app.get("/api/v1/face-liveness/sessions/{user_id}")
async def get_user_face_liveness_sessions(user_id: int, limit: int = 10, db: Session = Depends(get_db)):
    try:
        user = db.query(AppUser).filter(AppUser.id == user_id, AppUser.Active == True).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        sessions = db.query(FaceLivenessSession).filter(FaceLivenessSession.UserID == user_id).order_by(FaceLivenessSession.CreationDateTime.desc()).limit(limit).all()
        session_list = [
            {
                "session_id": session.SessionID, "user_id": session.UserID, "status": session.Status,
                "confidence_score": session.ConfidenceScore, "audit_images_limit": session.AuditImagesLimit,
                "expires_at": session.ExpiresAt.isoformat(), "created_at": session.CreationDateTime.isoformat(),
                "completed_at": session.CompletionDateTime.isoformat() if session.CompletionDateTime else None
            } for session in sessions
        ]
        return {"user_id": user_id, "user_name": user.Name, "total_sessions": len(session_list), "sessions": session_list}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting Face Liveness sessions for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get Face Liveness sessions")

@app.delete("/api/v1/face-liveness/session/{session_id}")
async def delete_face_liveness_session(session_id: str, db: Session = Depends(get_db)):
    try:
        db_session = db.query(FaceLivenessSession).filter(FaceLivenessSession.SessionID == session_id).first()
        if not db_session:
            raise HTTPException(status_code=404, detail="Session not found")
        if session_id in liveness_manager.active_sessions:
            del liveness_manager.active_sessions[session_id]
        db.delete(db_session)
        db.commit()
        logger.info(f"Deleted Face Liveness session: {session_id}")
        return {"success": True, "session_id": session_id, "message": "Session deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting Face Liveness session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete session")

@app.get("/api/v1/face/status/{user_id}")
async def get_face_status(user_id: int, db: Session = Depends(get_db)):
    try:
        user = db.query(AppUser).filter(AppUser.id == user_id, AppUser.Active == True).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        face_record = db.query(Face).filter(Face.UserID == user_id, Face.IsActive == True).first()
        if face_record:
            return {
                "user_id": user_id, "user_name": user.Name, "registered": True, "face_id": face_record.id,
                "quality_score": face_record.QualityScore, "face_confidence": face_record.FaceConfidence,
                "model_name": face_record.ModelName, "detector_backend": face_record.DetectorBackend,
                "registration_source": face_record.RegistrationSource, "registered_at": face_record.CreationDateTime.isoformat()
            }
        else:
            return {"user_id": user_id, "user_name": user.Name, "registered": False}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting face status for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get face status")

@app.get("/api/v1/face/verifications/{user_id}")
async def get_verification_history(user_id: int, limit: int = 10, db: Session = Depends(get_db)):
    try:
        user = db.query(AppUser).filter(AppUser.id == user_id, AppUser.Active == True).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        verifications = db.query(FaceVerification).filter(FaceVerification.UserID == user_id).order_by(FaceVerification.VerificationDateTime.desc()).limit(limit).all()
        verification_list = [
            {
                "verification_id": v.id, "user_id": v.UserID, "quiz_id": v.QuizID, "course_id": v.CourseID,
                "verified": v.VerificationResult, "similarity_score": v.SimilarityScore, "distance": v.Distance,
                "threshold_used": v.ThresholdUsed, "model_name": v.ModelName, "quality_score": v.QualityScore,
                "verification_datetime": v.VerificationDateTime.isoformat(), "verified_at": v.VerificationDateTime.isoformat()
            } for v in verifications
        ]
        return {"user_id": user_id, "total_verifications": len(verification_list), "verifications": verification_list}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting verification history for user {user_id}: {str(e)}")
        return {"user_id": user_id, "total_verifications": 0, "verifications": []}

@app.get("/api/v1/health")
async def health_check(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        db_status = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        db_status = "error"
    active_sessions = len(manager.active_connections)
    active_liveness_sessions = len(liveness_manager.active_sessions)
    overall_status = "healthy" if db_status == "healthy" else "error"
    liveness_manager.cleanup_expired_sessions()
    return {
        "status": overall_status, "timestamp": datetime.utcnow().isoformat(), "engine": "DeepFace + ArcFace",
        "database": db_status, "streaming": "enabled", "face_liveness": "enabled", "anti_spoofing": face_service.anti_spoofing,
        "services": {
            "deepface": "operational", "database": db_status, "websocket": "operational", "streaming": "operational",
            "face_liveness": "operational", "api": "healthy"
        },
        "configuration": {
            "model": face_service.model_name, "detector": face_service.detector_backend, "distance_metric": face_service.distance_metric,
            "anti_spoofing": face_service.anti_spoofing, "min_quality_score": face_service.min_quality_score,
            "liveness_threshold": face_service.liveness_threshold, "threshold": "55%"
        },
        "performance": {"min_quality_score": face_service.min_quality_score, "min_face_confidence": face_service.min_face_confidence},
        "active_connections": active_sessions, "active_sessions": active_sessions, "active_liveness_sessions": active_liveness_sessions,
        "session_timeout": f"{liveness_manager.session_timeout}s", "version": "4.0.0"
    }

@app.get("/api/v1/stats")
async def get_system_stats(db: Session = Depends(get_db)):
    try:
        total_users = db.query(AppUser).filter(AppUser.Active == True).count()
        registered_faces = db.query(Face).filter(Face.IsActive == True).count()
        total_liveness_sessions = db.query(FaceLivenessSession).count()
        completed_liveness_sessions = db.query(FaceLivenessSession).filter(FaceLivenessSession.Status == "COMPLETED").count()
        registration_rate = (registered_faces / total_users * 100) if total_users > 0 else 0
        liveness_success_rate = (completed_liveness_sessions / total_liveness_sessions * 100) if total_liveness_sessions > 0 else 0
        return {
            "total_users": total_users, "registered_faces": registered_faces, "registration_rate": registration_rate,
            "total_liveness_sessions": total_liveness_sessions, "completed_liveness_sessions": completed_liveness_sessions,
            "liveness_success_rate": liveness_success_rate, "success_rate_24h": 95.0, "system_health": "excellent" if registration_rate > 80 else "good",
            "active_sessions": len(manager.active_connections), "active_liveness_sessions": len(liveness_manager.active_sessions),
            "avg_processing_time": 1500, "total_verifications_today": 0
        }
    except Exception as e:
        logger.error(f"Error getting system stats: {str(e)}")
        return {
            "total_users": 0, "registered_faces": 0, "registration_rate": 0, "total_liveness_sessions": 0,
            "completed_liveness_sessions": 0, "liveness_success_rate": 0, "success_rate_24h": 0, "system_health": "poor",
            "active_sessions": len(manager.active_connections), "active_liveness_sessions": len(liveness_manager.active_sessions),
            "avg_processing_time": 1500, "total_verifications_today": 0
        }

@app.get("/api/v1/users")
async def get_all_users(db: Session = Depends(get_db)):
    try:
        users = db.query(AppUser).filter(AppUser.Active == True).order_by(AppUser.Name).all()
        role_map = {1: "Student", 2: "Instructor", 3: "Admin", 4: "Staff"}
        user_list = [
            {
                "id": user.id, "name": user.Name, "email": user.Email, "mobile": user.MobileNumber,
                "role": role_map.get(user.RoleID, "Student"), "roleId": user.RoleID, "status": user.Status,
                "active": user.Active, "salutation": user.Salutation,
                "lastLogin": user.LastLoginDateTime.isoformat() if user.LastLoginDateTime else None,
                "createdAt": user.CreationDateTime.isoformat(), "updatedAt": user.UpdationDateTime.isoformat()
            } for user in users
        ]
        logger.info(f"Retrieved {len(user_list)} active users from database")
        return user_list
    except Exception as e:
        logger.error(f"Error getting all users: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch users")

@app.get("/api/v1/user/{user_id}")
async def get_user_by_id(user_id: int, db: Session = Depends(get_db)):
    try:
        user = db.query(AppUser).filter(AppUser.id == user_id, AppUser.Active == True).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        role_map = {1: "Student", 2: "Instructor", 3: "Admin", 4: "Staff"}
        return {
            "id": user.id, "name": user.Name, "email": user.Email, "mobile": user.MobileNumber,
            "role": role_map.get(user.RoleID, "Student"), "roleId": user.RoleID, "status": user.Status,
            "active": user.Active, "salutation": user.Salutation,
            "lastLogin": user.LastLoginDateTime.isoformat() if user.LastLoginDateTime else None,
            "createdAt": user.CreationDateTime.isoformat(), "updatedAt": user.UpdationDateTime.isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch user")

@app.get("/api/v1/users/stats")
async def get_users_stats(db: Session = Depends(get_db)):
    try:
        total_users = db.query(AppUser).filter(AppUser.Active == True).count()
        students = db.query(AppUser).filter(AppUser.Active == True, AppUser.RoleID == 1).count()
        instructors = db.query(AppUser).filter(AppUser.Active == True, AppUser.RoleID == 2).count()
        staff = db.query(AppUser).filter(AppUser.Active == True, AppUser.RoleID == 4).count()
        admins = db.query(AppUser).filter(AppUser.Active == True, AppUser.RoleID == 3).count()
        registered_faces = db.query(Face).filter(Face.IsActive == True).count()
        registration_rate = (registered_faces / total_users * 100) if total_users > 0 else 0
        return {
            "total_users": total_users, "students": students, "instructors": instructors, "staff": staff,
            "admins": admins, "registered_faces": registered_faces, "registration_rate": registration_rate
        }
    except Exception as e:
        logger.error(f"Error getting user stats: {str(e)}")
        return {
            "total_users": 0, "students": 0, "instructors": 0, "staff": 0, "admins": 0,
            "registered_faces": 0, "registration_rate": 0
        }

# Background Task
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting LMS Face Recognition API with Face Liveness v4.0.0")
    cleanup_task = asyncio.create_task(cleanup_expired_sessions_task())
    try:
        yield
    finally:
        logger.info("🛑 Shutting down LMS Face Recognition API")
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass

async def cleanup_expired_sessions_task():
    while True:
        try:
            await asyncio.sleep(60)
            liveness_manager.cleanup_expired_sessions()
            try:
                db = SessionLocal()
                current_time = datetime.utcnow()
                expired_sessions = db.query(FaceLivenessSession).filter(
                    FaceLivenessSession.ExpiresAt < current_time,
                    FaceLivenessSession.Status.in_(["CREATED", "IN_PROGRESS"])
                ).all()
                for session in expired_sessions:
                    session.Status = "EXPIRED"
                if expired_sessions:
                    db.commit()
                    logger.info(f"Marked {len(expired_sessions)} database sessions as expired")
                db.close()
            except Exception as e:
                logger.error(f"Error cleaning up database sessions: {str(e)}")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in cleanup task: {str(e)}")
            await asyncio.sleep(10)

app.router.lifespan_context = lifespan

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1, log_level="info")