#!/usr/bin/env python3
"""
FastAPI Application for AI Interview Platform
Cloud-ready API for interview management with audio recording capabilities
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from dotenv import load_dotenv
except ImportError as e:
    print(f"FastAPI dependencies missing: {e}")
    print("Please install: pip install fastapi uvicorn python-multipart")
    sys.exit(1)

# Load environment variables
env_path = Path(__file__).parent.parent.parent.parent.parent / ".env"
load_dotenv(env_path)

# Initialize FastAPI app
app = FastAPI(
    title="AI Interview Platform API",
    description="Cloud-ready API for conducting AI-powered interviews with audio recording",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for web client compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global application state
app_state = {
    "start_time": datetime.now(),
    "interviews_conducted": 0,
    "active_sessions": 0,
    "audio_recordings_created": 0
}


@app.get("/health")
async def health_check():
    """Health check endpoint for deployment monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": (datetime.now() - app_state["start_time"]).total_seconds(),
        "service": "AI Interview Platform",
        "version": "1.0.0"
    }


@app.get("/")
async def root():
    """Root endpoint with basic information"""
    return {
        "message": "Welcome to AI Interview Platform API",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/stats")
async def get_stats():
    """Get basic application statistics"""
    return {
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": (datetime.now() - app_state["start_time"]).total_seconds(),
        "statistics": app_state
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    print("🚀 AI Interview Platform API starting up...")
    print(f"📅 Started at: {app_state['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    print("✅ FastAPI application ready")
    
    # Verify environment variables
    required_env_vars = ["DEEPGRAM_API_KEY", "OPENAI_API_KEY"]
    missing_vars = []
    
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️  Warning: Missing environment variables: {missing_vars}")
        print("🔧 Some features may not work without proper API keys")
    else:
        print("✅ All required environment variables found")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    print("🛑 AI Interview Platform API shutting down...")
    print(f"📊 Final statistics: {app_state}")
    print("✅ Shutdown complete")


if __name__ == "__main__":
    import uvicorn
    
    print("🔧 Starting AI Interview Platform API in development mode...")
    print("📝 For production deployment, use: uvicorn api:app --host 0.0.0.0 --port 8000")
    
    # Development server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
