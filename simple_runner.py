#!/usr/bin/env python3
"""
Simplified Traffic Violation Detection Runner
Works without heavy dependencies for demonstration
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

def create_mock_data():
    """Create mock violation data for demonstration"""
    mock_violations = [
        {
            "id": 1,
            "type": "no_helmet",
            "severity": "medium",
            "plate_number": "KA01AB1234",
            "vehicle_type": "motorcycle",
            "speed": None,
            "speed_limit": 40,
            "timestamp": datetime.now().isoformat(),
            "location": "Main Street Intersection",
            "confidence": 0.85
        },
        {
            "id": 2,
            "type": "mobile_usage",
            "severity": "high", 
            "plate_number": "KA02CD5678",
            "vehicle_type": "car",
            "speed": None,
            "speed_limit": 50,
            "timestamp": datetime.now().isoformat(),
            "location": "Main Street Intersection",
            "confidence": 0.92
        },
        {
            "id": 3,
            "type": "overspeed",
            "severity": "high",
            "plate_number": "KA03EF9012",
            "vehicle_type": "car", 
            "speed": 75,
            "speed_limit": 50,
            "timestamp": datetime.now().isoformat(),
            "location": "Main Street Intersection",
            "confidence": 0.88
        },
        {
            "id": 4,
            "type": "traffic_violation",
            "severity": "critical",
            "plate_number": "KA04GH3456",
            "vehicle_type": "motorcycle",
            "speed": None,
            "speed_limit": 40,
            "timestamp": datetime.now().isoformat(),
            "location": "Main Street Intersection", 
            "confidence": 0.91
        }
    ]
    
    # Create output directory and save mock data
    os.makedirs("demo_output", exist_ok=True)
    with open("demo_output/violations_log.json", "w") as f:
        json.dump(mock_violations, f, indent=2)
    
    print("✅ Mock violation data created")
    return mock_violations

def main():
    """Main runner function"""
    print("🚦 Traffic Violation Detection System")
    print("=" * 50)
    
    # Check project structure
    required_files = [
        "dashboard/index.html",
        "simple_server.py",
        "README.md"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ Project structure verified")
    
    # Create mock data for demonstration
    violations = create_mock_data()
    print(f"📊 Created {len(violations)} sample violations")
    
    # Show project status
    print("\n📋 Project Status:")
    print("   ✅ Dashboard available")
    print("   ✅ API server ready") 
    print("   ✅ Sample data loaded")
    print("   ✅ Ready to run")
    
    print("\n🚀 Starting the system...")
    print("📊 Dashboard will be available at: http://localhost:8081/dashboard")
    print("🔧 API docs at: http://localhost:8081/docs")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ System ready! Run 'python simple_server.py' to start the server.")
    else:
        print("\n❌ System setup failed!")