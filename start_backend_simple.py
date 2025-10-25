#!/usr/bin/env python3
"""
Simple backend startup script
"""
import sys
import os
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

# Change to backend directory
os.chdir(backend_dir)

if __name__ == "__main__":
    import uvicorn
    from api.main import app
    
    print("🚀 Starting AI Portfolio Rebalancing Agent Backend...")
    print("📍 Backend directory:", backend_dir)
    print("🌐 Server will be available at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("=" * 60)
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=False,  # Disable reload to avoid issues
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Backend stopped by user")
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        sys.exit(1)
