#!/usr/bin/env python3
"""
Simple FastAPI Server for Traffic Violation Detection
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import List, Dict
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Simple mode - no database dependencies
USE_REAL_DB = False
db_manager = None
print("📝 Running in simple mode with mock data")

# Create FastAPI app
app = FastAPI(title="Traffic Violation Detection API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load real violation data from demo output
def load_real_violations():
    """Load real violations from demo output"""
    try:
        with open('demo_output/violations_log.json', 'r') as f:
            violations = json.load(f)
        
        # Convert to dashboard format
        formatted_violations = []
        for i, violation in enumerate(violations):
            formatted_violations.append({
                "id": i + 1,
                "type": violation['type'],
                "plate_number": violation.get('plate_number', 'Unknown'),
                "speed": violation.get('speed', 'N/A'),
                "speed_limit": violation.get('speed_limit', 50),
                "timestamp": violation['timestamp'],
                "location": violation.get('location', 'Traffic Camera'),
                "confidence": violation['confidence']
            })
        return formatted_violations
    except Exception as e:
        print(f"Error loading real violations: {e}")
        return []

# Load real violations
real_violations = load_real_violations()

@app.get("/")
async def root():
    return {"message": "Traffic Violation Detection API is running!"}

@app.get("/stats")
async def get_stats():
    """Get violation statistics"""
    # Use real data
    if real_violations:
        # Count violation types
        violation_types = {}
        for violation in real_violations:
            vtype = violation['type']
            if vtype == 'no_helmet':
                violation_types['helmet'] = violation_types.get('helmet', 0) + 1
            elif vtype == 'mobile_usage':
                violation_types['phone'] = violation_types.get('phone', 0) + 1
            elif vtype == 'overspeed':
                violation_types['speed'] = violation_types.get('speed', 0) + 1
            elif vtype == 'traffic_violation':
                violation_types['red_light'] = violation_types.get('red_light', 0) + 1
        
        return {
            "total_violations": len(real_violations),
            "recent_violations_24h": len(real_violations),
            "violation_types": violation_types
        }
    
    # Fallback if no real data
    return {
        "total_violations": 0,
        "recent_violations_24h": 0,
        "violation_types": {
            "helmet": 0,
            "speed": 0,
            "phone": 0,
            "red_light": 0
        }
    }

@app.get("/violations")
async def get_violations(limit: int = 50):
    """Get violations list"""
    # Use real data
    if real_violations:
        return {
            "violations": real_violations[:limit],
            "total": len(real_violations)
        }
    
    # Fallback if no real data
    return {
        "violations": [],
        "total": 0
    }

@app.get("/dashboard")
async def dashboard():
    """Serve the dashboard"""
    return FileResponse("dashboard/index.html")

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """Upload and process video for violations"""
    return {
        "message": "Video upload feature temporarily disabled in simple mode",
        "filename": file.filename,
        "status": "info",
        "note": "Please run the full system locally for video processing"
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Simple Traffic Violation Detection Server...")
    print("📊 Dashboard: http://localhost:8000/dashboard")
    print("🔧 API Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8081)


