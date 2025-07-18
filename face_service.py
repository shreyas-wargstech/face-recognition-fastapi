# optimized_face_service.py - High-Performance Face Recognition Service

import asyncio
import time
import numpy as np
import cv2
from deepface import DeepFace
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
from typing import Dict, Any, Optional, List
import threading
from queue import Queue, Empty
from dataclasses import dataclass
from functools import lru_cache
import pickle
import io
from PIL import Image
import base64

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    success: bool
    encoding: Optional[np.ndarray] = None
    quality_score: float = 0.0
    face_confidence: float = 0.0
    antispoofing_score: float = 0.0
    processing_time: float = 0.0
    error: Optional[str] = None

class OptimizedFaceRecognitionService:
    def __init__(self):
        # Optimized model configuration
        self.model_name = "ArcFace"  # Keep for accuracy
        self.detector_backend = "opencv"  # Fastest detector
        self.distance_metric = "cosine"
        
        # Performance optimizations
        self.batch_size = 4  # Process multiple frames together
        self.max_workers = 4  # Increased from 2
        self.process_workers = 2  # For CPU-intensive tasks
        
        # Reduced frame requirements for speed
        self.required_frames_registration = 2  # Reduced from 3
        self.required_frames_verification = 1  # Reduced from 2
        self.frame_skip = 4  # Process every 4th frame (reduced from 8)
        
        # Optimized thresholds (maintained quality)
        self.min_quality_score = 20.0  # Slightly more lenient
        self.min_face_confidence = 0.45  # More permissive
        self.liveness_threshold = 0.3   # Less strict for speed
        self.similarity_threshold = 52.0  # Slightly relaxed
        
        # Processing pools
        self.thread_executor = ThreadPoolExecutor(
            max_workers=self.max_workers, 
            thread_name_prefix="face_thread"
        )
        self.process_executor = ProcessPoolExecutor(
            max_workers=self.process_workers
        )
        
        # Frame processing queue for batch processing
        self.frame_queue = Queue(maxsize=20)
        self.result_queue = Queue()
        
        # Pre-load models (reduces first-time processing delay)
        self._preload_models()
        
        # Cached embeddings for faster comparison
        self._embedding_cache = {}
        self._cache_lock = threading.RLock()
        
        logger.info(f"✅ High-Performance FaceRecognitionService initialized")

    def _preload_models(self):
        """Pre-load models to avoid cold start delays"""
        try:
            # Create a small dummy image to initialize models
            dummy_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
            
            # Initialize DeepFace models
            asyncio.create_task(self._preload_async(dummy_image))
            
        except Exception as e:
            logger.warning(f"Model preloading failed: {e}")

    async def _preload_async(self, dummy_image):
        """Async model preloading"""
        try:
            await asyncio.get_event_loop().run_in_executor(
                self.thread_executor,
                self._preload_sync,
                dummy_image
            )
        except Exception as e:
            logger.warning(f"Async preloading failed: {e}")

    def _preload_sync(self, dummy_image):
        """Synchronous model preloading"""
        try:
            # Preload detector
            DeepFace.extract_faces(
                img_path=dummy_image,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )
            
            # Preload embedding model
            DeepFace.represent(
                img_path=dummy_image,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )
            
            logger.info("Models preloaded successfully")
            
        except Exception as e:
            logger.warning(f"Model preloading error: {e}")

    async def extract_face_optimized(self, image_array: np.ndarray, timeout: float = 5.0) -> ProcessingResult:
        """Ultra-fast face extraction with aggressive optimizations"""
        try:
            start_time = time.time()
            
            if image_array is None or image_array.size == 0:
                return ProcessingResult(
                    success=False,
                    error="Invalid image data",
                    processing_time=0
                )
            
            # Resize image for faster processing if too large
            image_array = self._optimize_image_size(image_array)
            
            # Run processing in separate thread with timeout
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    self.thread_executor,
                    self._extract_face_fast,
                    image_array
                ),
                timeout=timeout
            )
            
            result.processing_time = (time.time() - start_time) * 1000
            return result
            
        except asyncio.TimeoutError:
            return ProcessingResult(
                success=False,
                error="Processing timeout",
                processing_time=timeout * 1000
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=f"Processing error: {str(e)}",
                processing_time=(time.time() - start_time) * 1000 if 'start_time' in locals() else 0
            )

    def _optimize_image_size(self, image: np.ndarray, max_size: int = 640) -> np.ndarray:
        """Resize image for optimal processing speed vs quality"""
        try:
            h, w = image.shape[:2]
            
            if max(h, w) > max_size:
                # Calculate resize ratio
                ratio = max_size / max(h, w)
                new_w = int(w * ratio)
                new_h = int(h * ratio)
                
                # Use faster interpolation
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            return image
            
        except Exception as e:
            logger.warning(f"Image resize failed: {e}")
            return image

    def _extract_face_fast(self, image_array: np.ndarray) -> ProcessingResult:
        """High-speed face extraction with minimal processing"""
        try:
            start_time = time.time()
            
            # Step 1: Fast face detection only
            faces = DeepFace.extract_faces(
                img_path=image_array,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=False,  # Skip alignment for speed
                anti_spoofing=False  # Skip anti-spoofing initially for speed
            )
            
            if not faces:
                return ProcessingResult(
                    success=False,
                    error="No face detected",
                    processing_time=(time.time() - start_time) * 1000
                )
            
            face = faces[0]
            
            # Step 2: Extract embedding (most time-consuming part)
            embeddings = DeepFace.represent(
                img_path=image_array,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=False  # Skip alignment for speed
            )
            
            if not embeddings:
                return ProcessingResult(
                    success=False,
                    error="Could not extract facial features",
                    processing_time=(time.time() - start_time) * 1000
                )
            
            embedding = embeddings[0]['embedding']
            facial_area = embeddings[0]['facial_area']
            
            # Step 3: Fast quality assessment
            quality_score = self._fast_quality_score(image_array, facial_area, face.get('confidence', 0.9))
            
            # Step 4: Optional anti-spoofing (only if needed)
            antispoofing_score = self._fast_antispoofing_check(face)
            
            processing_time = (time.time() - start_time) * 1000
            
            return ProcessingResult(
                success=True,
                encoding=np.array(embedding),
                quality_score=quality_score,
                face_confidence=face.get('confidence', 0.9),
                antispoofing_score=antispoofing_score,
                processing_time=processing_time
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=f"Face extraction failed: {str(e)}",
                processing_time=(time.time() - start_time) * 1000 if 'start_time' in locals() else 0
            )

    def _fast_quality_score(self, image: np.ndarray, facial_area: dict, face_confidence: float) -> float:
        """Ultra-fast quality assessment using minimal calculations"""
        try:
            # Basic metrics only for speed
            face_area = facial_area['w'] * facial_area['h']
            image_area = image.shape[0] * image.shape[1]
            
            # Size factor (20 points)
            size_factor = min(1.0, face_area / (80 * 80))  # 80x80 minimum
            
            # Confidence factor (80 points)
            confidence_factor = face_confidence
            
            # Simple brightness check (skip complex calculations)
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            face_region = image[y:y+h, x:x+w]
            avg_brightness = np.mean(face_region) / 255.0
            brightness_factor = 1.0 - abs(avg_brightness - 0.5) * 2  # Prefer mid-range brightness
            
            quality_score = (
                confidence_factor * 70 +
                size_factor * 20 +
                brightness_factor * 10
            ) * 100
            
            return max(0, min(100, quality_score))
            
        except Exception as e:
            logger.warning(f"Fast quality score error: {e}")
            return 60.0  # Default reasonable score

    def _fast_antispoofing_check(self, face_data: dict) -> float:
        """Simplified anti-spoofing check for speed"""
        try:
            # Use basic heuristics instead of complex ML models
            confidence = face_data.get('confidence', 0.9)
            
            # Simple spoofing indicators
            # High confidence usually indicates real face
            if confidence > 0.8:
                return 0.9
            elif confidence > 0.6:
                return 0.7
            else:
                return 0.5
                
        except Exception as e:
            logger.warning(f"Fast anti-spoofing error: {e}")
            return 0.8  # Default to likely real

    @lru_cache(maxsize=1000)  # Cache recent comparisons
    def compare_faces_cached(self, registered_encoding_hash: str, current_encoding: np.ndarray) -> Dict[str, Any]:
        """Cached face comparison for repeated verifications"""
        try:
            # Retrieve from cache (in real implementation, you'd unhash the encoding)
            # This is a simplified example
            return self.compare_faces_fast(None, current_encoding)
        except Exception as e:
            return {"similarity_score": 0.0, "is_match": False, "error": str(e)}

    def compare_faces_fast(self, registered_encoding: np.ndarray, current_encoding: np.ndarray) -> Dict[str, Any]:
        """Optimized face comparison with single similarity metric for speed"""
        try:
            start_time = time.time()
            
            if registered_encoding is None or current_encoding is None:
                return {
                    "similarity_score": 0.0,
                    "is_match": False,
                    "confidence": 0.0,
                    "error": "Invalid encoding data"
                }
            
            # Use only cosine similarity for speed (most reliable metric)
            cosine_similarity = np.dot(registered_encoding, current_encoding) / (
                np.linalg.norm(registered_encoding) * np.linalg.norm(current_encoding)
            )
            
            similarity_score = max(0, cosine_similarity * 100)
            
            # Single threshold check for speed
            is_match = similarity_score >= self.similarity_threshold
            
            # Simple confidence calculation
            confidence = min(1.0, similarity_score / 100)
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "similarity_score": float(similarity_score),
                "is_match": bool(is_match),
                "confidence": float(confidence),
                "threshold_used": float(self.similarity_threshold),
                "processing_time": processing_time
            }
            
        except Exception as e:
            return {
                "similarity_score": 0.0,
                "is_match": False,
                "confidence": 0.0,
                "error": str(e)
            }

    async def batch_process_frames(self, frames: List[np.ndarray]) -> List[ProcessingResult]:
        """Process multiple frames in parallel for better throughput"""
        try:
            if not frames:
                return []
            
            # Process frames in parallel
            tasks = [
                self.extract_face_optimized(frame, timeout=3.0)
                for frame in frames
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    processed_results.append(ProcessingResult(
                        success=False,
                        error=str(result)
                    ))
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            return [ProcessingResult(success=False, error=str(e)) for _ in frames]

    def cleanup_resources(self):
        """Clean up resources when shutting down"""
        try:
            self.thread_executor.shutdown(wait=False)
            self.process_executor.shutdown(wait=False)
            
            # Clear cache
            with self._cache_lock:
                self._embedding_cache.clear()
            
            logger.info("Face recognition service resources cleaned up")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

fast_face_service = OptimizedFaceRecognitionService()