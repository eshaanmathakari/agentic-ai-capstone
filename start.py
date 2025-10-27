#!/usr/bin/env python3
"""
AI Portfolio Rebalancing Agent - Startup Script
"""
import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import fastapi
        import streamlit
        import crewai
        import yfinance
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def setup_database():
    """Initialize the database"""
    try:
        backend_dir = Path(__file__).parent / "backend"
        os.chdir(backend_dir)
        
        # Initialize database
        subprocess.run([
            sys.executable, "-c", 
            "from database.connection import init_db; init_db()"
        ], check=True)
        
        print("✅ Database initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

def start_backend():
    """Start the backend server"""
    try:
        backend_dir = Path(__file__).parent / "backend"
        os.chdir(backend_dir)
        
        print("🚀 Starting AI Portfolio Rebalancing Agent Backend...")
        print("🌐 Server: http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/docs")
        
        subprocess.run([
            sys.executable, "-c",
            """
import uvicorn
from api.main import app
uvicorn.run(app, host='0.0.0.0', port=8000, log_level='info', access_log=True)
            """
        ])
    except KeyboardInterrupt:
        print("\n👋 Backend stopped by user")
    except Exception as e:
        print(f"❌ Backend startup failed: {e}")

def start_frontend():
    """Start the frontend application"""
    try:
        print("🎨 Starting Streamlit Frontend...")
        print("🌐 Frontend: http://localhost:8501")
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app/app.py"
        ])
    except KeyboardInterrupt:
        print("\n👋 Frontend stopped by user")
    except Exception as e:
        print(f"❌ Frontend startup failed: {e}")

def main():
    """Main startup function"""
    print("🤖 AI Portfolio Rebalancing Agent")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        return
    
    # Setup database
    if not setup_database():
        return
    
    print("\n🎯 Choose startup option:")
    print("1. Start Backend Only")
    print("2. Start Frontend Only") 
    print("3. Start Both (Backend + Frontend)")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        start_backend()
    elif choice == "2":
        start_frontend()
    elif choice == "3":
        print("\n⚠️  To start both, run in separate terminals:")
        print("Terminal 1: python start.py (choose option 1)")
        print("Terminal 2: python start.py (choose option 2)")
    elif choice == "4":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
