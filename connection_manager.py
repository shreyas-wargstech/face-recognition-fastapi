# optimized_websocket_handlers.py - High-Speed WebSocket Implementation

import asyncio
import json
import time
import uuid
import logging
from typing import Dict, Any, Optional
from fastapi import Depends, WebSocket, WebSocketDisconnect
from collections import deque
import numpy as np
from datetime import datetime

from sqlalchemy.orm import Session
logger = logging.getLogger(__name__)


class HighSpeedConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Dict] = {}
        self.processing_semaphores: Dict[str, asyncio.Semaphore] = {}
        self.frame_buffers: Dict[str, deque] = {}
        
        # Optimized settings for speed
        self.max_concurrent_processing = 3  # Process multiple frames simultaneously
        self.frame_buffer_size = 5  # Smaller buffer for faster processing
        self.heartbeat_interval = 60  # Less frequent heartbeats
        self.connection_timeout = 20  # Shorter timeout for faster failure detection
        
    async def connect_optimized(self, websocket: WebSocket, session_id: str, user_id: int, session_type: str):
        """Optimized connection setup"""
        await websocket.accept()
        
        self.active_connections[session_id] = {
            "websocket": websocket,
            "user_id": user_id,
            "session_type": session_type,
            "start_time": time.time(),
            "last_activity": time.time(),
            "frame_count": 0,
            "processed_count": 0,
            "processing_active": False,
            "results_buffer": deque(maxlen=10)
        }
        
        # Create processing semaphore for controlled concurrency
        self.processing_semaphores[session_id] = asyncio.Semaphore(self.max_concurrent_processing)
        self.frame_buffers[session_id] = deque(maxlen=self.frame_buffer_size)
        
        logger.info(f"🚀 High-speed WebSocket connected: {session_id}")

    async def send_fast(self, session_id: str, message: dict) -> bool:
        """Optimized message sending with minimal overhead"""
        if session_id not in self.active_connections:
            return False
        
        try:
            websocket = self.active_connections[session_id]["websocket"]
            
            # Use direct send without timeout for speed
            await websocket.send_text(json.dumps(message))
            self.active_connections[session_id]["last_activity"] = time.time()
            return True
            
        except Exception as e:
            logger.warning(f"Fast send failed for {session_id}: {e}")
            await self.disconnect_fast(session_id)
            return False

    async def disconnect_fast(self, session_id: str):
        """Fast cleanup without blocking operations"""
        try:
            if session_id in self.active_connections:
                # Clear buffers
                if session_id in self.frame_buffers:
                    self.frame_buffers[session_id].clear()
                    del self.frame_buffers[session_id]
                
                if session_id in self.processing_semaphores:
                    del self.processing_semaphores[session_id]
                
                del self.active_connections[session_id]
                
        except Exception as e:
            logger.error(f"Fast disconnect error: {e}")



# Initialize optimized services
speed_manager = HighSpeedConnectionManager()

