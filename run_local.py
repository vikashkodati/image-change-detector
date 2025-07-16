#!/usr/bin/env python3
"""
Local Development Runner for Image Change Detector
Simplified script for running the app locally
"""

import os
import sys
import subprocess
import time
import signal
from pathlib import Path

def check_env():
    """Check if .env file exists and has required variables"""
    env_file = Path("backend/.env")
    if not env_file.exists():
        print("⚠️  No .env file found in backend directory")
        print("Creating template .env file...")
        
        with open(env_file, "w") as f:
            f.write("# OpenAI API Key for GPT-4 Vision analysis\n")
            f.write("OPENAI_API_KEY=your_openai_api_key_here\n")
            f.write("\n# Local development settings\n")
            f.write("NODE_ENV=development\n")
        
        print("📝 Created backend/.env template")
        print("⚠️  Please add your OpenAI API key to backend/.env")
        return False
    
    # Check if API key is set
    with open(env_file, "r") as f:
        content = f.read()
        if "your_openai_api_key_here" in content:
            print("⚠️  Please update your OpenAI API key in backend/.env")
            return False
    
    print("✅ Environment configured")
    return True

def start_backend():
    """Start backend with proper environment"""
    print("🚀 Starting backend server...")
    
    # Use Python directly to avoid any UV issues
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path("backend/src").absolute())
    
    process = subprocess.Popen([
        sys.executable, "-m", "change_detector.server"
    ], cwd="backend", env=env)
    
    return process

def start_frontend():
    """Start frontend development server"""
    print("🚀 Starting frontend server...")
    
    # Set API URL for local development
    env = os.environ.copy()
    env["NEXT_PUBLIC_API_URL"] = "http://localhost:8000"
    
    process = subprocess.Popen([
        "npm", "run", "dev"
    ], cwd="frontend", env=env)
    
    return process

def main():
    """Main function for local development"""
    print("🌟 Image Change Detector - Local Development")
    print("=" * 50)
    
    # Check environment
    if not check_env():
        response = input("Continue without API key? (y/N): ").lower().strip()
        if response != 'y':
            print("Please configure your OpenAI API key and try again.")
            sys.exit(1)
    
    # Install dependencies if needed
    print("🔧 Installing dependencies...")
    
    # Backend
    if Path("backend/pyproject.toml").exists():
        try:
            subprocess.run(["pip", "install", "-e", "backend"], check=True, capture_output=True)
            print("✅ Backend dependencies installed")
        except subprocess.CalledProcessError:
            print("⚠️  Backend install failed, trying alternative method...")
            subprocess.run(["pip", "install", "fastapi", "uvicorn", "opencv-python", "numpy", "openai", "python-multipart", "fastmcp"], check=True)
    
    # Frontend
    if Path("frontend/package.json").exists():
        try:
            subprocess.run(["npm", "install"], cwd="frontend", check=True, capture_output=True)
            print("✅ Frontend dependencies installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Frontend install failed: {e}")
            sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎯 Starting services...")
    
    # Start backend
    backend_process = start_backend()
    
    # Wait for backend to start
    print("⏳ Waiting for backend to start...")
    time.sleep(3)
    
    # Start frontend
    frontend_process = start_frontend()
    
    print("\n" + "=" * 50)
    print("✅ Services running!")
    print("🖥️  Frontend: http://localhost:3000")
    print("🔧 Backend: http://localhost:8000") 
    print("📖 API Docs: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop all services")
    print("=" * 50)
    
    # Handle shutdown
    def signal_handler(sig, frame):
        print("\n\n🛑 Shutting down...")
        backend_process.terminate()
        frontend_process.terminate()
        
        # Wait for clean shutdown
        try:
            backend_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            backend_process.kill()
        
        try:
            frontend_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            frontend_process.kill()
        
        print("✅ Services stopped")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Keep running
    try:
        while True:
            time.sleep(1)
            # Check if processes are still running
            if backend_process.poll() is not None:
                print("❌ Backend process died")
                break
            if frontend_process.poll() is not None:
                print("❌ Frontend process died")
                break
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    main() 