#!/usr/bin/env python3
"""
Database Setup Script for Traffic Violation Detection System
This script helps set up PostgreSQL database and initialize tables.
"""

import os
import sys
import subprocess
from src.database import init_database, create_tables

def check_postgresql():
    """Check if PostgreSQL is installed and running"""
    try:
        # Check if psql is available
        result = subprocess.run(['psql', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ PostgreSQL is installed")
            return True
        else:
            print("✗ PostgreSQL is not installed")
            return False
    except FileNotFoundError:
        print("✗ PostgreSQL is not installed or not in PATH")
        return False

def create_database():
    """Create the traffic_violations database"""
    try:
        # Try to create database
        result = subprocess.run([
            'psql', '-U', 'postgres', '-h', 'localhost', 
            '-c', 'CREATE DATABASE traffic_violations;'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Database 'traffic_violations' created successfully")
            return True
        elif "already exists" in result.stderr:
            print("✓ Database 'traffic_violations' already exists")
            return True
        else:
            print(f"✗ Failed to create database: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error creating database: {e}")
        return False

def setup_environment():
    """Set up environment variables"""
    env_content = """# Database Configuration
DATABASE_URL=postgresql://postgres:password@localhost:5432/traffic_violations

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Model Configuration
MODEL_PATH=models/best.pt
DEVICE=auto

# Camera Configuration
DEFAULT_CAMERA_ID=CAM001
DEFAULT_LOCATION=Main Street Intersection

# Processing Configuration
CONFIDENCE_THRESHOLD=0.5
NMS_THRESHOLD=0.4
FRAME_SKIP=1

# Output Configuration
SAVE_VIOLATIONS=true
SAVE_PROCESSED_VIDEOS=true
OUTPUT_DIR=output
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✓ Environment file (.env) created")
        return True
    except Exception as e:
        print(f"✗ Failed to create .env file: {e}")
        return False

def main():
    """Main setup function"""
    print("🚗 Traffic Violation Detection - Database Setup")
    print("=" * 50)
    
    # Check PostgreSQL
    if not check_postgresql():
        print("\n📋 PostgreSQL Installation Instructions:")
        print("1. Download PostgreSQL from: https://www.postgresql.org/download/")
        print("2. Install with default settings")
        print("3. Set password for 'postgres' user to 'password'")
        print("4. Add PostgreSQL bin directory to PATH")
        print("5. Restart terminal and run this script again")
        return False
    
    # Create database
    if not create_database():
        print("\n📋 Manual Database Creation:")
        print("1. Open pgAdmin or psql")
        print("2. Create database: CREATE DATABASE traffic_violations;")
        print("3. Run this script again")
        return False
    
    # Setup environment
    if not setup_environment():
        return False
    
    # Initialize database tables
    try:
        print("\n🔧 Initializing database tables...")
        init_database()
        print("✓ Database setup completed successfully!")
        return True
    except Exception as e:
        print(f"✗ Database initialization failed: {e}")
        print("\n📋 Troubleshooting:")
        print("1. Make sure PostgreSQL is running")
        print("2. Check if user 'postgres' has password 'password'")
        print("3. Verify database 'traffic_violations' exists")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Database is ready! You can now run the application.")
        print("Next steps:")
        print("1. Run: python demo.py --mode api")
        print("2. Open: http://localhost:8000")
    else:
        print("\n❌ Database setup failed. Please check the instructions above.")
        sys.exit(1)
