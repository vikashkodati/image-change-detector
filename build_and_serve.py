#!/usr/bin/env python3
"""
Build and Serve Script for Image Change Detector
Starts both frontend and backend services as per architecture
"""

import os
import sys
import subprocess
import time
import signal
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    # Check if uv is installed
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        print("✅ UV package manager found")
    except subprocess.CalledProcessError:
        print("❌ UV not found. Please install UV first: https://docs.astral.sh/uv/")
        return False
    
    # Check if npm is installed
    try:
        subprocess.run(["npm", "--version"], check=True, capture_output=True)
        print("✅ NPM found")
    except subprocess.CalledProcessError:
        print("❌ NPM not found. Please install Node.js and NPM")
        return False
    
    return True

def setup_backend():
    """Setup backend dependencies"""
    print("\n🔧 Setting up backend...")
    
    backend_dir = Path("backend")
    if not backend_dir.exists():
        print("❌ Backend directory not found")
        return False
    
    # Change to backend directory and sync dependencies
    os.chdir(backend_dir)
    try:
        subprocess.run(["uv", "sync"], check=True)
        print("✅ Backend dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install backend dependencies: {e}")
        return False
    finally:
        os.chdir("..")
    
    return True

def setup_frontend():
    """Setup frontend dependencies"""
    print("\n🔧 Setting up frontend...")
    
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return False
    
    # Change to frontend directory and install dependencies
    os.chdir(frontend_dir)
    try:
        subprocess.run(["npm", "install"], check=True)
        print("✅ Frontend dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install frontend dependencies: {e}")
        return False
    finally:
        os.chdir("..")
    
    return True

def check_env_file():
    """Check if .env file exists in backend"""
    env_file = Path("backend/.env")
    if not env_file.exists():
        print("\n⚠️  No .env file found in backend directory")
        print("Creating template .env file...")
        
        with open(env_file, "w") as f:
            f.write("# OpenAI API Key for GPT-4 Vision analysis\n")
            f.write("OPENAI_API_KEY=your_openai_api_key_here\n")
        
        print("📝 Created backend/.env template")
        print("⚠️  Please add your OpenAI API key to backend/.env")
        return False
    
    print("✅ Environment file found")
    return True

def start_backend():
    """Start the FastMCP backend server"""
    print("\n🚀 Starting backend server...")
    
    backend_dir = Path("backend")
    os.chdir(backend_dir)
    
    # Start the main FastMCP server using UV to ensure dependencies are available
    # Set PYTHONPATH to ensure the module can be found
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path("src").absolute())
    
    process = subprocess.Popen([
        "uv", "run", "python", "-m", "change_detector.server"
    ], env=env)
    
    os.chdir("..")
    return process

def start_frontend():
    """Start the Next.js frontend"""
    print("\n🚀 Starting frontend server...")
    
    frontend_dir = Path("frontend")
    os.chdir(frontend_dir)
    
    # Start Next.js development server
    process = subprocess.Popen([
        "npm", "run", "dev"
    ])
    
    os.chdir("..")
    return process

def main():
    """Main function to build and serve the application"""
    print("🌟 Image Change Detector - Build and Serve")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Setup backend
    if not setup_backend():
        sys.exit(1)
    
    # Setup frontend
    if not setup_frontend():
        sys.exit(1)
    
    # Check environment
    env_ready = check_env_file()
    if not env_ready:
        print("\nPlease configure your .env file before continuing.")
        response = input("Continue anyway? (y/N): ").lower().strip()
        if response != 'y':
            sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎯 Starting services...")
    
    # Start backend
    backend_process = start_backend()
    
    # Wait a moment for backend to start
    print("⏳ Waiting for backend to start...")
    time.sleep(3)
    
    # Start frontend
    frontend_process = start_frontend()
    
    print("\n" + "=" * 50)
    print("✅ Services started!")
    print("🖥️  Frontend: http://localhost:3000")
    print("🔧 Backend: http://localhost:8000")
    print("📖 API Docs: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop all services")
    print("=" * 50)
    
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print("\n\n🛑 Shutting down services...")
        backend_process.terminate()
        frontend_process.terminate()
        
        # Wait for processes to terminate
        backend_process.wait()
        frontend_process.wait()
        
        print("✅ Services stopped")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Wait for processes
    try:
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    main()