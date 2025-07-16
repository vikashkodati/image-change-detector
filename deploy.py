#!/usr/bin/env python3
"""
Deployment Helper Script for Image Change Detector
Helps prepare and deploy the MCP-powered app to production
"""

import os
import subprocess
import sys

def run_command(cmd, cwd=None):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Command failed: {cmd}")
            print(f"Error: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False

def main():
    print("🚀 Image Change Detector - Deployment Helper")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("frontend") or not os.path.exists("backend"):
        print("❌ Please run this script from the root directory (cd1/)")
        sys.exit(1)
    
    print("📋 Pre-deployment checklist:")
    print()
    
    # Check if git is initialized
    if not os.path.exists(".git"):
        print("🔧 Initializing git repository...")
        if not run_command("git init"):
            sys.exit(1)
    
    # Check if .env exists
    backend_env = "backend/.env"
    if not os.path.exists(backend_env):
        print("⚠️  No backend/.env file found")
        print("📝 Creating template .env file...")
        
        with open(backend_env, "w") as f:
            f.write("# OpenAI API Key for GPT-4 Vision analysis\n")
            f.write("OPENAI_API_KEY=your_openai_api_key_here\n")
        
        print(f"✅ Created {backend_env}")
        print("⚠️  Please add your OpenAI API key to backend/.env before deploying")
        print()
    
    # Test local build
    print("🧪 Testing local build...")
    
    # Test backend dependencies
    print("  📦 Checking backend dependencies...")
    if not run_command("uv sync", cwd="backend"):
        print("❌ Backend dependency check failed")
        sys.exit(1)
    
    # Test frontend dependencies
    print("  📦 Checking frontend dependencies...")
    if not run_command("npm install", cwd="frontend"):
        print("❌ Frontend dependency check failed")
        sys.exit(1)
    
    # Test frontend build
    print("  🏗️  Testing frontend build...")
    if not run_command("npm run build", cwd="frontend"):
        print("❌ Frontend build failed")
        sys.exit(1)
    
    print("✅ All checks passed!")
    print()
    
    print("🌐 Ready for deployment!")
    print()
    print("📋 Next Steps:")
    print("1. 📤 Push to GitHub:")
    print("   git add .")
    print("   git commit -m 'Prepare for deployment'")
    print("   git push origin main")
    print()
    print("2. 🚂 Deploy Backend (Railway):")
    print("   • Go to https://railway.app")
    print("   • Click 'Deploy from GitHub repo'")
    print("   • Select your repository")
    print("   • Add environment variable: OPENAI_API_KEY=your_key")
    print("   • Copy the deployed URL")
    print()
    print("3. ⚡ Deploy Frontend (Vercel):")
    print("   • Go to https://vercel.com")
    print("   • Import your GitHub repository")
    print("   • Set root directory to 'frontend'")
    print("   • Add environment variable: NEXT_PUBLIC_API_URL=your_railway_url")
    print("   • Deploy!")
    print()
    print("🎉 Your MCP-powered Image Change Detector will be live!")
    print("📖 Full instructions in README.md")

if __name__ == "__main__":
    main() 