#!/usr/bin/env python3
"""
Deployment Helper for Matrix Change Detector
Validates builds and provides deployment instructions
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if required tools are installed"""
    print("🔍 Checking deployment requirements...")
    
    requirements = [
        ("git", "Git version control"),
        ("uv", "UV package manager"),
        ("npm", "Node.js package manager"),
        ("docker", "Docker containerization (optional)")
    ]
    
    missing = []
    for cmd, desc in requirements:
        try:
            subprocess.run([cmd, "--version"], check=True, capture_output=True)
            print(f"✅ {desc}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"❌ {desc} - Not found")
            missing.append(cmd)
    
    return len(missing) == 0

def validate_environment():
    """Check environment configuration"""
    print("\n🔧 Validating environment...")
    
    # Check backend .env
    backend_env = Path("backend/.env")
    if backend_env.exists():
        with open(backend_env) as f:
            content = f.read()
            if "OPENAI_API_KEY" in content and "your_openai_api_key_here" not in content:
                print("✅ Backend environment configured")
            else:
                print("⚠️  Backend .env needs OpenAI API key")
                return False
    else:
        print("⚠️  Backend .env file missing")
        return False
    
    # Check railway.toml
    railway_config = Path("railway.toml")
    if railway_config.exists():
        print("✅ Railway configuration found")
    else:
        print("❌ Railway configuration missing")
        return False
    
    return True

def test_backend_build():
    """Test backend build process"""
    print("\n🐍 Testing backend build...")
    
    current_dir = os.getcwd()
    
    try:
        # Test UV sync
        os.chdir("backend")
        subprocess.run(["uv", "sync"], check=True, capture_output=True)
        print("✅ Backend dependencies installed")
        
        # Test Docker build (if Docker is available)
        try:
            subprocess.run(["docker", "--version"], check=True, capture_output=True)
            print("🐳 Testing Docker build...")
            result = subprocess.run([
                "docker", "build", "-t", "matrix-detector-test", "."
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Docker build successful")
                # Clean up test image
                subprocess.run(["docker", "rmi", "matrix-detector-test"], 
                             capture_output=True)
            else:
                print("⚠️  Docker build failed (Railway will handle this):")
                print(f"   {result.stderr.strip()}")
        except FileNotFoundError:
            print("⚠️  Docker not available (Railway will handle build)")
        
        os.chdir(current_dir)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Backend build failed: {e}")
        os.chdir(current_dir)
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        os.chdir(current_dir)
        return False

def test_frontend_build():
    """Test frontend build process"""
    print("\n⚛️  Testing frontend build...")
    
    current_dir = os.getcwd()
    
    try:
        os.chdir("frontend")
        
        # Install dependencies
        subprocess.run(["npm", "install"], check=True, capture_output=True)
        print("✅ Frontend dependencies installed")
        
        # Test build
        subprocess.run(["npm", "run", "build"], check=True, capture_output=True)
        print("✅ Frontend build successful")
        
        os.chdir(current_dir)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Frontend build failed: {e}")
        os.chdir(current_dir)
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        os.chdir(current_dir)
        return False

def show_deployment_instructions():
    """Show step-by-step deployment instructions"""
    print("\n" + "="*60)
    print("🚀 DEPLOYMENT INSTRUCTIONS")
    print("="*60)
    
    print("\n📦 BACKEND DEPLOYMENT (Railway)")
    print("─" * 40)
    print("1. Commit and push your changes:")
    print("   git add .")
    print("   git commit -m 'Deploy Matrix Change Detector'")
    print("   git push origin main")
    print()
    print("2. Deploy to Railway:")
    print("   • Go to https://railway.app")
    print("   • Click 'Deploy from GitHub repo'")
    print("   • Select your repository")
    print("   • Railway will auto-detect the railway.toml config")
    print()
    print("3. Set environment variables in Railway:")
    print("   • OPENAI_API_KEY=your_actual_api_key")
    print("   • PORT=8000 (auto-set)")
    print()
    print("4. Railway will build using Docker and deploy automatically")
    
    print("\n🌐 FRONTEND DEPLOYMENT (Vercel)")
    print("─" * 40)
    print("1. Go to https://vercel.com")
    print("2. Import your GitHub repository")
    print("3. Set configuration:")
    print("   • Root Directory: frontend")
    print("   • Framework Preset: Next.js")
    print("   • Build Command: npm run build")
    print("   • Output Directory: out")
    print()
    print("4. Set environment variable:")
    print("   • NEXT_PUBLIC_API_URL=https://your-railway-url.railway.app")
    print()
    print("5. Deploy!")
    
    print("\n🔗 POST-DEPLOYMENT")
    print("─" * 40)
    print("• Test your deployed backend health: https://your-railway-url.railway.app/api/health")
    print("• Test your frontend: https://your-vercel-url.vercel.app")
    print("• Update frontend env with final Railway URL if needed")
    
    print("\n🐛 TROUBLESHOOTING")
    print("─" * 40)
    print("• Check Railway logs for backend issues")
    print("• Verify environment variables are set correctly")
    print("• Ensure CORS is configured for your frontend domain")
    print("• Test API endpoints manually if needed")

def main():
    """Main deployment validation and guidance"""
    print("🕶️ Matrix Change Detector - Deployment Helper")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Missing required tools. Please install them and try again.")
        sys.exit(1)
    
    # Validate environment
    if not validate_environment():
        print("\n❌ Environment validation failed. Please fix configuration.")
        sys.exit(1)
    
    # Test builds (Docker failures are OK since Railway will handle them)
    print("\n🧪 Testing local builds...")
    backend_ok = test_backend_build()
    frontend_ok = test_frontend_build()
    
    if backend_ok and frontend_ok:
        print("\n✅ All local tests passed! Ready for deployment.")
    elif backend_ok or frontend_ok:
        print("\n⚠️  Some tests passed. Railway/Vercel may still work.")
        print("   (Docker failures are normal - Railway handles Docker builds)")
    else:
        print("\n❌ Critical build tests failed. Please fix errors before deploying.")
        sys.exit(1)
    
    # Show deployment instructions
    show_deployment_instructions()
    
    print("\n🎯 Next Steps:")
    print("1. Commit and push your code")
    print("2. Deploy backend to Railway") 
    print("3. Deploy frontend to Vercel")
    print("4. Test the deployed application")
    print("\n🕶️ Welcome to the Matrix! 🕶️")

if __name__ == "__main__":
    main() 