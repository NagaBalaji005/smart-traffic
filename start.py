#!/usr/bin/env python3
"""
Traffic Violation Detection System - Main Startup Script
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🚀 Traffic Violation Detection System")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists('models/traffic_violations_best.pt'):
        print("❌ Error: Trained model not found!")
        print("💡 Make sure you're in the project root directory")
        return
    
    print("✅ Trained model found")
    
    # Check if dashboard exists
    if not os.path.exists('dashboard/index.html'):
        print("❌ Error: Dashboard not found!")
        return
    
    print("✅ Dashboard found")
    
    # Start the server
    print("\n🌐 Starting web server...")
    print("📊 Dashboard will be available at: http://localhost:8081/dashboard")
    print("🔧 API docs at: http://localhost:8081/docs")
    print("\n⏹️ Press Ctrl+C to stop the server")
    
    try:
        # Start the simple server
        subprocess.run([sys.executable, "simple_server.py"])
    except KeyboardInterrupt:
        print("\n👋 Server stopped. Goodbye!")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    main()
