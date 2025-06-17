# main_fixed.py - Fixed WebSocket Face Recognition with proper timeout and connection handling

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

# imports for websockets
from contextlib import asynccontextmanager
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import psutil
import resource

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

class OptimizedConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Dict] = {}
        self.connection_timestamps: Dict[str, datetime] = {}
        self.processing_locks: Dict[str, asyncio.Lock] = {}
        self.cleanup_task = None
        self.max_connection_time = 300  # 5 minutes max
        self.heartbeat_interval = 30  # 30 seconds
        
    async def connect(self, websocket: WebSocket, session_id: str, user_id: int, session_type: str):
        await websocket.accept()
        
        # Store connection with optimized settings
        self.active_connections[session_id] = {
            "websocket": websocket,
            "user_id": user_id,
            "session_type": session_type,
            "frame_buffer": deque(maxlen=10),  # Reduced buffer size
            "frame_count": 0,
            "processed_count": 0,
            "quality_scores": deque(maxlen=20),
            "last_activity": time.time(),
            "start_time": time.time(),
            "processing": False,
            "last_heartbeat": time.time(),
            "timeout_warnings": 0
        }
        
        self.connection_timestamps[session_id] = datetime.utcnow()
        self.processing_locks[session_id] = asyncio.Lock()
        
        # Start heartbeat for this connection
        asyncio.create_task(self._heartbeat_loop(session_id))
        
        logger.info(f"🔗 WebSocket connected: {session_id} for user {user_id} ({session_type})")
        
        # Start cleanup task if not running
        if self.cleanup_task is None:
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            try:
                # Clean up resources
                connection = self.active_connections[session_id]
                if connection.get("frame_buffer"):
                    connection["frame_buffer"].clear()
                
                del self.active_connections[session_id]
                del self.connection_timestamps[session_id]
                if session_id in self.processing_locks:
                    del self.processing_locks[session_id]
                
                logger.info(f"❌ WebSocket disconnected: {session_id}")
                
                # Force garbage collection
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error during disconnect cleanup: {str(e)}")
    
    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            try:
                websocket = self.active_connections[session_id]["websocket"]
                await asyncio.wait_for(
                    websocket.send_text(json.dumps(message)), 
                    timeout=5.0
                )
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
        """Force disconnect a problematic connection"""
        if session_id in self.active_connections:
            try:
                websocket = self.active_connections[session_id]["websocket"]
                await websocket.close(code=1000, reason="Server timeout")
            except:
                pass
            finally:
                await self.disconnect(session_id)
    
    async def _heartbeat_loop(self, session_id: str):
        """Send periodic heartbeat to keep connection alive"""
        while session_id in self.active_connections:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                if session_id not in self.active_connections:
                    break
                
                connection = self.active_connections[session_id]
                current_time = time.time()
                
                # Check if connection is still active
                if current_time - connection.get("last_activity", 0) > 60:  # 1 minute silence
                    success = await self.send_message(session_id, {
                        "type": "heartbeat",
                        "timestamp": current_time,
                        "session_time": current_time - connection["start_time"]
                    })
                    
                    if not success:
                        break
                        
            except Exception as e:
                logger.error(f"Heartbeat error for {session_id}: {str(e)}")
                break
    
    async def _cleanup_loop(self):
        """Clean up stale connections and manage resources"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                current_time = time.time()
                stale_sessions = []
                
                for session_id, connection in self.active_connections.items():
                    # Check for various timeout conditions
                    session_age = current_time - connection["start_time"]
                    last_activity = current_time - connection.get("last_activity", current_time)
                    
                    if (session_age > self.max_connection_time or 
                        last_activity > 120 or  # 2 minutes of inactivity
                        connection.get("timeout_warnings", 0) > 3):
                        
                        stale_sessions.append(session_id)
                
                # Clean up stale sessions
                for session_id in stale_sessions:
                    logger.warning(f"Cleaning up stale session: {session_id}")
                    await self.force_disconnect(session_id)
                
                # Log resource usage
                if len(self.active_connections) > 0:
                    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
                    logger.info(f"Active connections: {len(self.active_connections)}, Memory: {memory_usage:.1f}MB")
                
            except Exception as e:
                logger.error(f"Cleanup loop error: {str(e)}")
                await asyncio.sleep(10)

# Enhanced Face Recognition Service with better error handling
class OptimizedFaceRecognitionService:
    def __init__(self):
        self.model_name = "ArcFace"
        self.detector_backend = "opencv"  # Faster than retinaface
        self.distance_metric = "cosine"
        self.anti_spoofing = True
        
        # Optimized thresholds for better success rate
        self.min_quality_score = 25.0  # More lenient
        self.min_face_confidence = 0.5  # More permissive
        self.liveness_threshold = 0.4   # Less strict
        
        # Reduced frame requirements for faster processing
        self.required_frames = 3  # Minimum frames
        self.max_frames = 8       # Maximum frames
        self.frame_skip = 8       # Process every 8th frame
        
        # Thread pool with limited workers
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="face_processing")
        self.processing_timeout = 30  # 30 seconds max per frame
        
        logger.info(f"✅ Optimized FaceRecognitionService initialized")

    async def extract_face_async(self, image_array, timeout: float = 15.0) -> Dict[str, Any]:
        """Async wrapper for face extraction with timeout"""
        try:
            loop = asyncio.get_event_loop()
            
            # Run in thread pool with timeout
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    self.executor, 
                    self._extract_face_sync, 
                    image_array
                ),
                timeout=timeout
            )
            
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"Face extraction timeout after {timeout}s")
            return {
                "success": False,
                "error": "Processing timeout - please try again",
                "spoofing_detected": False,
                "processing_time": timeout * 1000
            }
        except Exception as e:
            logger.error(f"Face extraction error: {str(e)}")
            return {
                "success": False,
                "error": f"Processing error: {str(e)}",
                "spoofing_detected": False,
                "processing_time": 0
            }

    def _extract_face_sync(self, image_array) -> Dict[str, Any]:
        """Synchronous face extraction - runs in thread pool"""
        try:
            start_time = time.time()
            
            # Simplified face extraction for speed
            faces = DeepFace.extract_faces(
                img_path=image_array,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=True,
                anti_spoofing=self.anti_spoofing
            )
            
            if not faces:
                return {
                    "success": False,
                    "error": "No face detected",
                    "spoofing_detected": False,
                    "processing_time": (time.time() - start_time) * 1000
                }
            
            # Use first face for speed
            face = faces[0]
            
            # Quick spoofing check
            if not face.get('is_real', True):
                return {
                    "success": False,
                    "error": "Please use your real face",
                    "spoofing_detected": True,
                    "antispoofing_score": face.get('antispoof_score', 0.0),
                    "processing_time": (time.time() - start_time) * 1000
                }
            
            # Extract embedding
            embeddings = DeepFace.represent(
                img_path=image_array,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=True
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
            
            # Quick quality calculation
            quality_score = self._quick_quality_score(image_array, facial_area, face.get('confidence', 0.9))
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "success": True,
                "encoding": np.array(embedding),
                "facial_area": facial_area,
                "quality_score": quality_score,
                "face_confidence": face.get('confidence', 0.9),
                "antispoofing_score": face.get('antispoof_score', 1.0),
                "is_real": face.get('is_real', True),
                "spoofing_detected": False,
                "model_name": self.model_name,
                "processing_time": processing_time
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Processing error: {str(e)}",
                "spoofing_detected": False,
                "processing_time": (time.time() - start_time) * 1000
            }

    def _quick_quality_score(self, image: np.ndarray, facial_area: dict, face_confidence: float) -> float:
        """Quick quality assessment for faster processing"""
        try:
            # Basic quality checks only
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            
            # Size factor
            face_area = w * h
            size_factor = min(1.0, face_area / (100 * 100))  # 100x100 minimum
            
            # Basic score
            quality_score = (
                face_confidence * 60 +    # Detection confidence (60%)
                size_factor * 40          # Size factor (40%)
            ) * 100
            
            return max(0, min(100, quality_score))
            
        except Exception:
            return 50.0  # Default score
    
    def compare_faces_with_verification(self, registered_encoding: np.ndarray, current_encoding: np.ndarray) -> Dict[str, Any]:
        """Enhanced face comparison with multiple similarity metrics for better verification"""
        try:
            start_time = time.time()
            
            # Calculate multiple similarity metrics
            # 1. Cosine similarity (primary)
            cosine_similarity = np.dot(registered_encoding, current_encoding) / (
                np.linalg.norm(registered_encoding) * np.linalg.norm(current_encoding)
            )
            
            # 2. Euclidean distance
            euclidean_distance = np.linalg.norm(registered_encoding - current_encoding)
            
            # 3. Manhattan distance (L1)
            manhattan_distance = np.sum(np.abs(registered_encoding - current_encoding))
            
            # Convert cosine similarity to percentage
            similarity_score = max(0, cosine_similarity * 100)
            
            # Enhanced threshold logic - use multiple metrics
            cosine_threshold = 0.55  # Relaxed from 0.68
            euclidean_threshold = 1.2  # Relaxed threshold
            
            # Multi-criteria matching
            cosine_match = cosine_similarity >= cosine_threshold
            euclidean_match = euclidean_distance <= euclidean_threshold
            similarity_match = similarity_score >= 55.0
            
            # Final decision: any two criteria must pass
            is_match = sum([cosine_match, euclidean_match, similarity_match]) >= 2
            
            # Calculate confidence score
            confidence = (similarity_score / 100) * 0.7 + (1 - min(euclidean_distance / 2.0, 1.0)) * 0.3
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "similarity_score": float(similarity_score),
                "cosine_similarity": float(cosine_similarity),
                "euclidean_distance": float(euclidean_distance),
                "manhattan_distance": float(manhattan_distance),
                "is_match": bool(is_match),
                "confidence": float(confidence),
                "cosine_match": bool(cosine_match),
                "euclidean_match": bool(euclidean_match),
                "similarity_match": bool(similarity_match),
                "threshold_used": float(cosine_threshold),
                "model_name": self.model_name,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error comparing faces for verification: {str(e)}")
            return {
                "similarity_score": 0.0,
                "is_match": False,
                "confidence": 0.0,
                "error": str(e),
                "processing_time": 0
            }

# Initialize optimized services
manager = OptimizedConnectionManager()
face_service = OptimizedFaceRecognitionService()

@app.websocket("/ws/face-registration/{user_id}")
async def optimized_face_registration_stream(websocket: WebSocket, user_id: int, db: Session = Depends(get_db)):
    """Optimized real-time face registration with proper timeout handling"""
    session_id = f"reg_{user_id}_{uuid.uuid4().hex[:8]}"
    
    try:
        # Check if user exists
        user = db.query(AppUser).filter(AppUser.id == user_id, AppUser.Active == True).first()
        if not user:
            await websocket.close(code=4004, reason="User not found")
            return
        
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
        frame_processing_times = []
        last_db_commit = time.time()
        
        while True:
            try:
                # Receive frame with timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                message_data = json.loads(data)
                
                if message_data.get("type") == "frame":
                    connection = manager.active_connections.get(session_id)
                    if not connection:
                        break
                    
                    # Check if already processing
                    async with manager.processing_locks[session_id]:
                        if connection.get("processing", False):
                            continue  # Skip if still processing previous frame
                        
                        connection["processing"] = True
                    
                    try:
                        frame_data = message_data["frame"]
                        connection["frame_count"] += 1
                        
                        # Skip frames for performance (process every Nth frame)
                        if connection["frame_count"] % face_service.frame_skip != 0:
                            continue
                        
                        # Decode frame
                        frame = decode_base64_frame(frame_data)
                        if frame is None:
                            await manager.send_message(session_id, {
                                "type": "error",
                                "message": "Invalid frame data"
                            })
                            continue
                        
                        # Process frame asynchronously with timeout
                        process_start = time.time()
                        result = await face_service.extract_face_async(frame, timeout=10.0)
                        process_time = time.time() - process_start
                        
                        frame_processing_times.append(process_time)
                        connection["processed_count"] += 1
                        
                        # Handle spoofing detection
                        if result["spoofing_detected"]:
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
                                "required_frames": face_service.required_frames,
                                "processing_time": process_time
                            })
                            continue
                        
                        # More lenient quality check
                        if result["quality_score"] < face_service.min_quality_score:
                            await manager.send_message(session_id, {
                                "type": "frame_processed",
                                "success": False,
                                "message": f"Frame quality: {result['quality_score']:.1f}/100 (need >{face_service.min_quality_score})",
                                "quality_score": result["quality_score"],
                                "frames_collected": len(best_frames),
                                "required_frames": face_service.required_frames,
                                "processing_time": process_time
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
                        
                        await manager.send_message(session_id, {
                            "type": "frame_processed",
                            "success": True,
                            "quality_score": result["quality_score"],
                            "antispoofing_score": result["antispoofing_score"],
                            "face_confidence": result["face_confidence"],
                            "frames_collected": len(best_frames),
                            "required_frames": face_service.required_frames,
                            "processing_time": process_time,
                            "message": f"Good frame! ({len(best_frames)}/{face_service.required_frames})"
                        })
                        
                        # Check if we have enough frames
                        if len(best_frames) >= face_service.required_frames:
                            # Select best frame
                            best_frame = max(best_frames, key=lambda x: x["quality_score"])
                            
                            # Calculate averages
                            avg_quality = sum(quality_scores) / len(quality_scores)
                            avg_antispoofing = sum(f["antispoofing_score"] for f in best_frames) / len(best_frames)
                            avg_processing_time = sum(frame_processing_times) / len(frame_processing_times)
                            
                            try:
                                # Database operations with timeout
                                async with asyncio.timeout(10.0):
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
                                        RegistrationSource="stream_v2"
                                    )
                                    
                                    db.add(new_face)
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
                                        "avg_processing_time": avg_processing_time,
                                        "message": "Face registration completed successfully!"
                                    })
                                    
                                    logger.info(f"✅ Face registered for user {user_id} in {avg_processing_time:.2f}s avg")
                                    break
                                    
                            except asyncio.TimeoutError:
                                await manager.send_message(session_id, {
                                    "type": "error",
                                    "message": "Database timeout - please try again"
                                })
                                break
                            except Exception as e:
                                logger.error(f"Database error: {str(e)}")
                                await manager.send_message(session_id, {
                                    "type": "error", 
                                    "message": f"Registration failed: {str(e)}"
                                })
                                break
                        
                        # Stop if we've processed too many frames
                        if len(best_frames) >= face_service.max_frames:
                            break
                            
                    finally:
                        connection["processing"] = False
                
                elif message_data.get("type") == "stop":
                    break
                elif message_data.get("type") == "ping":
                    await manager.send_message(session_id, {"type": "pong"})
                    
            except asyncio.TimeoutError:
                logger.warning(f"WebSocket receive timeout for {session_id}")
                await manager.send_message(session_id, {
                    "type": "timeout_warning",
                    "message": "Connection timeout - please check your network"
                })
                
                connection = manager.active_connections.get(session_id)
                if connection:
                    connection["timeout_warnings"] = connection.get("timeout_warnings", 0) + 1
                    if connection["timeout_warnings"] > 2:
                        break
                        
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Frame processing error: {str(e)}")
                await manager.send_message(session_id, {
                    "type": "error",
                    "message": f"Processing error: {str(e)}"
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected normally: {session_id}")
    except Exception as e:
        logger.error(f"Registration stream error for {session_id}: {str(e)}")
        logger.error(traceback.format_exc())
        try:
            await manager.send_message(session_id, {
                "type": "error",
                "message": f"Registration failed: {str(e)}"
            })
        except:
            pass
    finally:
        await manager.disconnect(session_id)

@app.websocket("/ws/face-verification/{user_id}")
async def optimized_face_verification_stream(
    websocket: WebSocket, 
    user_id: int, 
    quiz_id: str = None, 
    course_id: str = None,
    db: Session = Depends(get_db)
):
    """Optimized real-time face verification with timeout handling and relaxed thresholds"""
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
        
        await manager.connect(websocket, session_id, user_id, "verification")
        
        # Decrypt registered face encoding
        try:
            registered_encoding = decrypt_face_encoding(face_registration.FaceEmbedding)
        except Exception as e:
            logger.error(f"Failed to decrypt face encoding for user {user_id}: {str(e)}")
            await manager.send_message(session_id, {
                "type": "error",
                "message": "Failed to load registered face data"
            })
            return
        
        # Send initial status with relaxed requirements
        await manager.send_message(session_id, {
            "type": "connected",
            "session_id": session_id,
            "user_id": user_id,
            "user_name": user.Name,
            "quiz_id": quiz_id,
            "course_id": course_id,
            "required_frames": 2,  # Only 2 frames needed for verification
            "similarity_threshold": 55.0,  # Relaxed threshold
            "message": "Ready for identity verification. Please look at the camera."
        })
        
        verification_frames = []
        quality_scores = []
        similarity_scores = []
        antispoofing_scores = []
        frame_processing_times = []
        verification_attempts = 0
        max_verification_attempts = 5  # Allow multiple attempts
        
        while True:
            try:
                # Receive frame with timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                message_data = json.loads(data)
                
                if message_data.get("type") == "frame":
                    connection = manager.active_connections.get(session_id)
                    if not connection:
                        break
                    
                    # Check if already processing
                    async with manager.processing_locks[session_id]:
                        if connection.get("processing", False):
                            continue  # Skip if still processing previous frame
                        
                        connection["processing"] = True
                    
                    try:
                        frame_data = message_data["frame"]
                        connection["frame_count"] += 1
                        verification_attempts += 1
                        
                        # Skip frames for performance (process every 8th frame for verification)
                        if connection["frame_count"] % 8 != 0:
                            continue
                        
                        # Decode frame
                        frame = decode_base64_frame(frame_data)
                        if frame is None:
                            await manager.send_message(session_id, {
                                "type": "error",
                                "message": "Invalid frame data"
                            })
                            continue
                        
                        # Process frame asynchronously with shorter timeout for verification
                        process_start = time.time()
                        result = await face_service.extract_face_async(frame, timeout=8.0)
                        process_time = time.time() - process_start
                        
                        frame_processing_times.append(process_time)
                        connection["processed_count"] += 1
                        
                        # Handle spoofing detection (more lenient for verification)
                        if result["spoofing_detected"]:
                            await manager.send_message(session_id, {
                                "type": "spoofing_detected",
                                "message": "Please ensure you're using your real face",
                                "antispoofing_score": result.get("antispoofing_score", 0.0),
                                "can_retry": verification_attempts < max_verification_attempts
                            })
                            
                            # Don't break immediately, allow retry
                            if verification_attempts >= max_verification_attempts:
                                break
                            continue
                        
                        if not result["success"]:
                            await manager.send_message(session_id, {
                                "type": "frame_processed",
                                "success": False,
                                "message": result["error"],
                                "frames_collected": len(verification_frames),
                                "required_frames": 2,
                                "processing_time": process_time,
                                "attempts_remaining": max_verification_attempts - verification_attempts
                            })
                            
                            # Allow retry for verification
                            if verification_attempts >= max_verification_attempts:
                                break
                            continue
                        
                        # More lenient quality check for verification
                        if result["quality_score"] < 10.0:  # Very low threshold
                            await manager.send_message(session_id, {
                                "type": "frame_processed", 
                                "success": False,
                                "message": f"Frame quality: {result['quality_score']:.1f}/100 (minimum: 10)",
                                "quality_score": result["quality_score"],
                                "frames_collected": len(verification_frames),
                                "required_frames": 2,
                                "processing_time": process_time,
                                "attempts_remaining": max_verification_attempts - verification_attempts
                            })
                            
                            if verification_attempts >= max_verification_attempts:
                                break
                            continue
                        
                        # Compare with registered face - async operation
                        try:
                            comparison_start = time.time()
                            comparison_result = await asyncio.wait_for(
                                asyncio.get_event_loop().run_in_executor(
                                    face_service.executor,
                                    face_service.compare_faces_with_verification,
                                    registered_encoding,
                                    result["encoding"]
                                ),
                                timeout=5.0
                            )
                            comparison_time = time.time() - comparison_start
                            
                        except asyncio.TimeoutError:
                            logger.warning(f"Face comparison timeout for session {session_id}")
                            await manager.send_message(session_id, {
                                "type": "error",
                                "message": "Face comparison timeout - please try again"
                            })
                            continue
                        except Exception as e:
                            logger.error(f"Face comparison error: {str(e)}")
                            await manager.send_message(session_id, {
                                "type": "error",
                                "message": "Face comparison failed - please try again"
                            })
                            continue
                        
                        # Store verification frame with results
                        verification_frame_data = {
                            "quality_score": result["quality_score"],
                            "antispoofing_score": result["antispoofing_score"],
                            "similarity_score": comparison_result["similarity_score"],
                            "is_match": comparison_result["similarity_score"] >= 55.0,  # Relaxed threshold
                            "processing_time": process_time,
                            "comparison_time": comparison_time
                        }
                        
                        verification_frames.append(verification_frame_data)
                        quality_scores.append(result["quality_score"])
                        similarity_scores.append(comparison_result["similarity_score"])
                        antispoofing_scores.append(result["antispoofing_score"])
                        
                        await manager.send_message(session_id, {
                            "type": "frame_processed",
                            "success": True,
                            "quality_score": result["quality_score"],
                            "antispoofing_score": result["antispoofing_score"],
                            "similarity_score": comparison_result["similarity_score"],
                            "is_match": verification_frame_data["is_match"],
                            "frames_collected": len(verification_frames),
                            "required_frames": 2,
                            "processing_time": process_time,
                            "comparison_time": comparison_time,
                            "message": f"Frame verified! ({len(verification_frames)}/2) - Similarity: {comparison_result['similarity_score']:.1f}%"
                        })
                        
                        # Check if we have enough frames for verification (only 2 needed)
                        if len(verification_frames) >= 2:
                            # Calculate verification result with relaxed criteria
                            avg_quality = sum(quality_scores) / len(quality_scores)
                            avg_antispoofing = sum(antispoofing_scores) / len(antispoofing_scores)
                            avg_similarity = sum(similarity_scores) / len(similarity_scores)
                            max_similarity = max(similarity_scores)  # Use best similarity score
                            
                            # Count matching frames
                            matching_frames = sum(1 for frame in verification_frames if frame["is_match"])
                            match_ratio = matching_frames / len(verification_frames)
                            
                            # Relaxed verification decision - multiple criteria for better success rate
                            is_verified = (
                                # Primary criteria: high similarity OR good match ratio
                                (max_similarity >= 55.0 or avg_similarity >= 50.0) and
                                # Secondary criteria: reasonable antispoofing (very lenient)
                                avg_antispoofing >= 0.2 and
                                # Tertiary criteria: at least 1 matching frame OR good average
                                (matching_frames >= 1 or match_ratio >= 0.5)
                            )
                            
                            # Calculate additional metrics
                            avg_processing_time = sum(frame_processing_times) / len(frame_processing_times)
                            confidence_score = min(100, max_similarity + (match_ratio * 20))
                            
                            try:
                                # Database operations with timeout
                                async with asyncio.timeout(10.0):
                                    # Log verification attempt
                                    verification = FaceVerification(
                                        UserID=user_id,
                                        QuizID=quiz_id,
                                        CourseID=course_id,
                                        VerificationResult=is_verified,
                                        SimilarityScore=avg_similarity,
                                        Distance=1 - (avg_similarity / 100),
                                        ThresholdUsed=55.0,  # Document the relaxed threshold
                                        ModelName=face_service.model_name,
                                        DistanceMetric=face_service.distance_metric,
                                        ProcessingTime=sum(frame_processing_times),
                                        QualityScore=avg_quality
                                    )
                                    
                                    db.add(verification)
                                    
                                    # Update user's last login if verification successful
                                    if is_verified:
                                        user.LastLoginDateTime = datetime.utcnow()
                                    
                                    db.commit()
                                    db.refresh(verification)
                                    
                                    # Send comprehensive verification result
                                    await manager.send_message(session_id, {
                                        "type": "verification_complete",
                                        "success": True,
                                        "verification_id": verification.id,
                                        "user_id": user_id,
                                        "user_name": user.Name,
                                        "verified": is_verified,
                                        "similarity_score": avg_similarity,
                                        "max_similarity_score": max_similarity,
                                        "quality_score": avg_quality,
                                        "antispoofing_score": avg_antispoofing,
                                        "match_ratio": match_ratio,
                                        "confidence_score": confidence_score,
                                        "frames_processed": len(verification_frames),
                                        "avg_processing_time": avg_processing_time,
                                        "quiz_id": quiz_id,
                                        "course_id": course_id,
                                        "threshold_used": 55.0,
                                        "verification_method": "optimized_stream",
                                        "message": "Identity verified successfully!" if is_verified else "Identity verification failed - please try again"
                                    })
                                    
                                    if is_verified:
                                        logger.info(f"✅ Face verification PASSED for user {user_id} (similarity: {max_similarity:.1f}%, quiz: {quiz_id})")
                                    else:
                                        logger.warning(f"❌ Face verification FAILED for user {user_id} (similarity: {max_similarity:.1f}%, quiz: {quiz_id})")
                                    
                                    break
                                    
                            except asyncio.TimeoutError:
                                await manager.send_message(session_id, {
                                    "type": "error",
                                    "message": "Database timeout - please try again"
                                })
                                break
                            except Exception as e:
                                logger.error(f"Database error during verification: {str(e)}")
                                await manager.send_message(session_id, {
                                    "type": "error",
                                    "message": f"Verification failed: {str(e)}"
                                })
                                break
                        
                        # Stop if we've processed maximum frames or attempts
                        if len(verification_frames) >= 4 or verification_attempts >= max_verification_attempts:
                            # Attempt verification with whatever frames we have
                            if len(verification_frames) > 0:
                                # Force verification with available frames
                                best_similarity = max(similarity_scores) if similarity_scores else 0
                                is_verified = best_similarity >= 50.0  # Even more lenient final threshold
                                
                                await manager.send_message(session_id, {
                                    "type": "verification_complete",
                                    "success": True,
                                    "verified": is_verified,
                                    "similarity_score": best_similarity,
                                    "frames_processed": len(verification_frames),
                                    "message": "Verification completed with available frames" if is_verified else "Verification failed - insufficient similarity"
                                })
                            break
                            
                    finally:
                        connection["processing"] = False
                
                elif message_data.get("type") == "stop":
                    break
                elif message_data.get("type") == "ping":
                    await manager.send_message(session_id, {"type": "pong"})
                elif message_data.get("type") == "restart_verification":
                    # Allow restarting verification process
                    verification_frames.clear()
                    quality_scores.clear()
                    similarity_scores.clear()
                    antispoofing_scores.clear()
                    verification_attempts = 0
                    await manager.send_message(session_id, {
                        "type": "verification_restarted",
                        "message": "Verification restarted - please look at the camera"
                    })
                    
            except asyncio.TimeoutError:
                logger.warning(f"WebSocket receive timeout for verification session {session_id}")
                await manager.send_message(session_id, {
                    "type": "timeout_warning",
                    "message": "Connection timeout - please check your network",
                    "can_retry": verification_attempts < max_verification_attempts
                })
                
                connection = manager.active_connections.get(session_id)
                if connection:
                    connection["timeout_warnings"] = connection.get("timeout_warnings", 0) + 1
                    if connection["timeout_warnings"] > 3:
                        break
                        
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Frame processing error in verification: {str(e)}")
                await manager.send_message(session_id, {
                    "type": "error",
                    "message": f"Processing error: {str(e)}",
                    "can_retry": verification_attempts < max_verification_attempts
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected normally: {session_id}")
    except Exception as e:
        logger.error(f"Verification stream error for {session_id}: {str(e)}")
        logger.error(traceback.format_exc())
        try:
            await manager.send_message(session_id, {
                "type": "error",
                "message": f"Verification failed: {str(e)}"
            })
        except:
            pass
    finally:
        await manager.disconnect(session_id)

# Add application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting optimized face recognition service")
    yield
    # Shutdown
    logger.info("🛑 Shutting down face recognition service")
    if hasattr(face_service, 'executor'):
        face_service.executor.shutdown(wait=True)

# Apply lifespan to app
app = FastAPI(lifespan=lifespan)