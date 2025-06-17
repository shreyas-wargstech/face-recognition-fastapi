#!/usr/bin/env python3
"""
Health check script for Face Recognition Backend
Monitors system performance and connection status
"""

import asyncio
import aiohttp
import psutil
import time
import json
from datetime import datetime

async def check_api_health():
    """Check API health endpoint"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8000/api/v1/health', timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ API Health: {data.get('status', 'unknown')}")
                    return True
                else:
                    print(f"❌ API Health: HTTP {response.status}")
                    return False
    except Exception as e:
        print(f"❌ API Health: {e}")
        return False

def check_system_health():
    """Check system resource usage"""
    memory = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=1)
    
    print(f"💾 Memory: {memory.percent:.1f}% used ({memory.available // (1024**2):.0f}MB available)")
    print(f"🖥️  CPU: {cpu:.1f}% usage")
    
    # Check for concerning resource usage
    issues = []
    if memory.percent > 85:
        issues.append("High memory usage")
    if cpu > 90:
        issues.append("High CPU usage")
    if memory.available < 512 * 1024**2:  # Less than 512MB
        issues.append("Low available memory")
    
    if issues:
        print(f"⚠️  Issues: {', '.join(issues)}")
        return False
    else:
        print("✅ System resources healthy")
        return True

def check_port_status():
    """Check if backend port is open"""
    import socket
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', 8000))
        sock.close()
        
        if result == 0:
            print("✅ Backend port 8000 is open")
            return True
        else:
            print("❌ Backend port 8000 is not accessible")
            return False
    except Exception as e:
        print(f"❌ Port check failed: {e}")
        return False

async def main():
    """Main health check function"""
    print("🏥 Face Recognition Backend Health Check")
    print("=" * 40)
    print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run all health checks
    port_ok = check_port_status()
    system_ok = check_system_health()
    api_ok = await check_api_health() if port_ok else False
    
    print()
    print("📊 Health Summary:")
    print(f"   Port Status: {'✅' if port_ok else '❌'}")
    print(f"   System Resources: {'✅' if system_ok else '❌'}")
    print(f"   API Health: {'✅' if api_ok else '❌'}")
    
    overall_status = all([port_ok, system_ok, api_ok])
    print(f"   Overall: {'✅ HEALTHY' if overall_status else '❌ ISSUES DETECTED'}")
    
    return overall_status

if __name__ == "__main__":
    asyncio.run(main())
