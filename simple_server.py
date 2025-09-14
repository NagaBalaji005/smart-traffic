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

# Add src to path for database imports
sys.path.append('src')

# Try to import database manager
try:
    from database import DatabaseManager
    db_manager = DatabaseManager()
    print("✅ Database connected successfully")
    USE_REAL_DB = True
except Exception as e:
    print(f"⚠️ Database connection failed: {e}")
    print("📝 Using mock data instead")
    db_manager = None
    USE_REAL_DB = False

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
        import json
        with open('demo_output/violations_log.json', 'r') as f:
            violations = json.load(f)
        
        # Convert to dashboard format
        formatted_violations = []
        for i, violation in enumerate(violations[:50]):  # Limit to 50 for performance
            formatted_violations.append({
                "id": i + 1,
                "type": violation['type'],
                "plate_number": violation.get('plate_number', 'DEMO123'),
                "speed": 65 if violation['type'] == 'overspeed' else 45,
                "speed_limit": 50,
                "timestamp": violation['timestamp'],
                "location": "Traffic Camera",
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
    if USE_REAL_DB and db_manager:
        try:
            # Get real statistics from database
            stats = db_manager.get_violation_stats()
            return stats
        except Exception as e:
            print(f"Error getting stats from database: {e}")
            # Fallback to mock data
            pass
    
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
    if USE_REAL_DB and db_manager:
        try:
            # Get real violations from database
            violations = db_manager.get_recent_violations(limit=limit)
            return {
                "violations": violations,
                "total": len(violations)
            }
        except Exception as e:
            print(f"Error getting violations from database: {e}")
            # Fallback to mock data
            pass
    
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
    try:
        # Save uploaded file
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        print(f"🎬 Processing uploaded video: {file.filename}")
        
        # Process video using our video processor
        import subprocess
        import json
        
        # Run video processing
        output_dir = "processed_output"
        os.makedirs(output_dir, exist_ok=True)
        
        result = subprocess.run([
            sys.executable, "simple_video_processor.py", 
            file_path, "--output", output_dir
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        print(f"📊 Processing result: returncode={result.returncode}")
        print(f"📝 STDOUT: {result.stdout[-200:]}")  # Last 200 chars
        print(f"📝 STDERR: {result.stderr}")
        
        if result.returncode == 0:
            print("✅ Video processed successfully")
            
            # Load the processed violations
            violations_file = os.path.join(output_dir, "violations_log.json")
            if os.path.exists(violations_file):
                with open(violations_file, 'r') as f:
                    violations = json.load(f)
                
                print(f"📊 Loaded {len(violations)} violations from file")
                
                # Convert to dashboard format
                formatted_violations = []
                for i, violation in enumerate(violations[:50]):
                    # Skip if no meaningful data
                    if not violation.get('plate_number') and not violation.get('speed'):
                        continue
                        
                    # Get unique violations only (avoid duplicates)
                    violation_key = f"{violation['type']}_{violation.get('track_id', i)}"
                    
                    formatted_violations.append({
                        "id": i + 1,
                        "type": violation['type'],
                        "plate_number": violation.get('plate_number', 'Unknown'),
                        "speed": violation.get('speed', 'N/A'),
                        "speed_limit": violation.get('speed_limit', 50),
                        "timestamp": violation['timestamp'],
                        "location": "Uploaded Video",
                        "confidence": violation['confidence'],
                        "track_id": violation.get('track_id', 'N/A')
                        "track_id": violation.get('track_id', 'N/A')
                    })
                
                # Remove duplicates
                unique_violations = []
                seen_violations = set()
                
                for violation in formatted_violations:
                    key = f"{violation['type']}_{violation['track_id']}"
                    if key not in seen_violations:
                        seen_violations.add(key)
                        unique_violations.append(violation)
                
                # Remove duplicates based on track_id and violation type
                unique_violations = []
                seen_violations = set()
                
                for violation in formatted_violations:
                    key = f"{violation['type']}_{violation['track_id']}"
                    if key not in seen_violations:
                        seen_violations.add(key)
                        unique_violations.append(violation)
                
                return unique_violations
                # Update global violations
                global real_violations
                real_violations = unique_violations
                
                print(f"✅ Updated global violations: {len(unique_violations)} violations")
                
                return {
                    "message": "Video processed successfully",
                    "filename": file.filename,
                    "status": "completed",
                    "violations_detected": len(unique_violations),
                    "output_video": f"{output_dir}/detected_{Path(file.filename).stem}.mp4"
                }
            else:
                print("⚠️ No violations file found")
                return {
                    "message": "Video processed but no violations detected",
                    "filename": file.filename,
                    "status": "completed",
                    "violations_detected": 0
                }
        else:
            print(f"❌ Video processing failed: {result.stderr}")
            return {
                "message": "Video processing failed",
                "filename": file.filename,
                "status": "error",
                "error": result.stderr
            }
        
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Simple Traffic Violation Detection Server...")
    print("📊 Dashboard: http://localhost:8000/dashboard")
    print("🔧 API Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8081)


