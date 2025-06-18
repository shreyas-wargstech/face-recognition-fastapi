#!/usr/bin/env python3
"""
Optimized startup script for Face Recognition Backend
Applies performance optimizations and monitoring
"""

import os
import sys
import asyncio
import logging
import psutil
import gc
from pathlib import Path

# Set optimized environment variables
os.environ.setdefault('PYTHONUNBUFFERED', '1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def optimize_python_settings():
    """Apply Python performance optimizations"""
    # Garbage collection optimization
    gc.set_threshold(700, 10, 10)  # More aggressive GC
    
    # Set recursion limit
    sys.setrecursionlimit(1500)
    
    logger.info("🚀 Python optimizations applied")

def check_system_resources():
    """Check system resources and warn if low"""
    memory = psutil.virtual_memory()
    cpu_count = psutil.cpu_count()
    
    logger.info(f"💾 Memory: {memory.total // (1024**3)}GB total, {memory.available // (1024**3)}GB available")
    logger.info(f"🖥️  CPU cores: {cpu_count}")
    
    if memory.percent > 80:
        logger.warning("⚠️  High memory usage detected")
    
    if memory.available < 1024**3:  # Less than 1GB
        logger.warning("⚠️  Low available memory (< 1GB)")

def main():
    """Main startup function"""
    print("🚀 Starting Optimized Face Recognition Backend")
    print("=" * 50)
    
    # Apply optimizations
    optimize_python_settings()
    check_system_resources()
    
    # Load environment
    try:
        from dotenv import load_dotenv
        load_dotenv()
        logger.info("✅ Environment variables loaded")
    except ImportError:
        logger.warning("⚠️  python-dotenv not found, using system environment")
    
    # Import and start the app
    try:
        import uvicorn
        
        # Optimized uvicorn configuration
        config = {
            'app': 'main:app',
            'host': '0.0.0.0',
            'port': 8000,
            'workers': 1,  # Single worker for WebSocket support
            'log_level': 'info',
            'access_log': True,
            'loop': 'asyncio',
            'ws_ping_interval': 20,
            'ws_ping_timeout': 10,
            'timeout_keep_alive': 30,
            'limit_max_requests': 1000,
            'limit_concurrency': 100
        }
        
        logger.info("🌐 Starting server with optimized configuration...")
        logger.info(f"🔗 Server will be available at http://localhost:8000")
        logger.info("📡 WebSocket endpoints:")
        logger.info("   - /ws/face-registration/{user_id}")
        logger.info("   - /ws/face-verification/{user_id}")
        
        uvicorn.run(**config)
        
    except ImportError:
        logger.error("❌ uvicorn not found. Install with: pip install uvicorn")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
