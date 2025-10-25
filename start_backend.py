#!/usr/bin/env python3
"""
Backend startup script with proper import handling
"""
import sys
import os
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

# Change to backend directory
os.chdir(backend_dir)

# Import and run the application
if __name__ == "__main__":
    import uvicorn
    from api.main import app
    
    print("🚀 Starting AI Portfolio Rebalancing Agent Backend...")
    print("📍 Backend directory:", backend_dir)
    print("🌐 Server will be available at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
