# Matrix Change Detector - Development Knowledge Base

## 🕶️ Project Overview
**Matrix Change Detector** is a cyberpunk-themed satellite image analysis application that combines **FastMCP (Model Context Protocol)** backend with a **Matrix movie-inspired frontend**. The system provides AI-powered change detection using OpenCV computer vision and GPT-4 Vision analysis.

### 🎯 Current Status: **PRODUCTION READY**
- **Theme**: Complete Matrix cyberpunk interface with digital rain
- **Backend**: FastMCP server with MCP tools + REST API bridge
- **Frontend**: Next.js with authentic Matrix UI elements
- **Deployment**: Railway (backend) + Vercel (frontend) ready
- **Testing**: Comprehensive local and production testing

## 🏗️ Architecture Implementation

### **System Design**
```
Matrix Rain Background (Canvas) 
    ↓
Cyberpunk UI Components (React + Tailwind)
    ↓
HTTP REST API (Base64 Images)
    ↓
FastAPI Bridge → MCP Tools
    ↓
OpenCV + GPT-4 Vision Processing
```

### **Technology Stack Details**

#### 🖥️ **Frontend Matrix Stack**
- **Next.js 14** - React framework with TypeScript
- **Matrix Rain Component** - Canvas-based falling katakana animation
- **Tailwind CSS** - Custom Matrix theme with cyberpunk classes
- **Shadcn/ui** - Modified components with Matrix styling
- **Custom CSS** - Glitch effects, scanlines, glow animations

#### 🔧 **Backend Stack**
- **FastMCP v0.1.0+** - MCP protocol implementation
- **FastAPI** - REST API bridge for frontend integration
- **OpenCV-Python 4.8+** - Computer vision processing
- **OpenAI GPT-4 Vision** - AI semantic analysis
- **UV Package Manager** - Python dependency management
- **Uvicorn** - ASGI server

#### 🌐 **Deployment Stack**
- **Railway** - Backend container deployment with auto-scaling
- **Vercel** - Frontend static site with edge CDN
- **Docker** - Containerized backend with multi-stage builds
- **Environment Variables** - Secure API key management

## 🎨 Matrix Theme Implementation

### **Visual Elements Implemented**
```css
/* Core Matrix Classes */
.matrix-text         /* Green text with glow */
.matrix-glow         /* Neon border effects */
.matrix-border       /* Semi-transparent green borders */
.matrix-button       /* Cyberpunk button styling */
.matrix-card         /* Dark glass card backgrounds */
.matrix-pulse        /* Breathing animations */
.matrix-scanline     /* Moving scanline effects */
.matrix-glitch       /* RGB offset glitch text */
```

### **Matrix Rain Animation**
- **Characters**: Japanese katakana + numbers + symbols
- **Animation**: 35ms refresh rate for smooth movement
- **Depth**: Multiple green shades for 3D effect
- **Performance**: Optimized Canvas rendering
- **Responsive**: Auto-resizes with window

### **Cyberpunk UI Features**
- **Color Scheme**: Matrix green (#00FF00) variants on black
- **Typography**: Monospace fonts (Courier New, Monaco, Menlo)
- **Animations**: Glitch, pulse, scanline, and glow effects
- **Terminology**: Cyberpunk language ("NEURAL NETWORK", "SURVEILLANCE DATA")

## 🔧 MCP Tools Architecture

### **Available MCP Tools**
```python
@mcp.tool()
async def detect_image_changes(before_image_base64: str, after_image_base64: str)
    """OpenCV-based pixel difference detection"""
    
@mcp.tool()
async def analyze_changes_with_ai(before_image_base64: str, after_image_base64: str, change_results: dict)
    """GPT-4 Vision semantic analysis"""
    
@mcp.tool()
async def answer_question_about_changes(question: str, before_image_base64: str, after_image_base64: str)
    """Interactive Q&A about detected changes"""
    
@mcp.tool()
async def health_check()
    """Server health monitoring"""
```

### **REST API Bridge**
```python
# FastAPI endpoints that call MCP tools
@app.post("/api/detect-changes")
@app.post("/api/analyze-changes") 
@app.post("/api/answer-question")
@app.get("/api/health")
```

### **Change Detection Algorithm**
1. **Input Processing**: Base64 → OpenCV format
2. **Preprocessing**: Resize images to match dimensions
3. **Difference Calculation**: Grayscale absolute difference
4. **Thresholding**: Binary mask with threshold=30
5. **Contour Detection**: Find changed regions
6. **Statistics**: Calculate change percentage and metrics
7. **Visualization**: Generate colored change mask

## 🚀 Build & Deployment

### **Local Development Scripts**

#### `build_and_serve.py` - Main Development Script
```python
# Features:
- Dependency checking (UV, NPM)
- Backend dependency installation (uv sync)
- Frontend dependency installation (npm install)
- Environment file validation
- Concurrent server startup
- Graceful shutdown handling
```

#### `run_local.py` - Alternative Runner
```python
# Features:
- Simplified dependency management
- Direct Python path configuration
- Environment validation with fallbacks
- Error recovery mechanisms
```

#### `deploy.py` - Deployment Helper
```python
# Features:
- Build validation
- Deployment readiness checks
- Environment configuration verification
- Step-by-step deployment instructions
```

### **Production Deployment Configuration**

#### Railway Backend (`railway.toml`)
```toml
[build]
builder = "DOCKERFILE"

[deploy]
startCommand = "uvicorn src.change_detector.server:app --host 0.0.0.0 --port $PORT"

[env]
PYTHONPATH = "/app/src"
```

#### Vercel Frontend (`next.config.mjs`)
```javascript
/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  trailingSlash: true,
  images: {
    unoptimized: true
  }
};
```

## 🧪 Testing Strategy

### **Local Testing Checklist**
```bash
# 1. Full Application Test
python build_and_serve.py
# ✅ Both services start
# ✅ Frontend accessible at localhost:3000
# ✅ Backend accessible at localhost:8000
# ✅ API docs at localhost:8000/docs

# 2. Matrix UI Validation
# ✅ Digital rain animation visible
# ✅ Green cyberpunk color scheme
# ✅ Glitch effect on main title
# ✅ Scanline animations on interface
# ✅ Glow effects on borders/buttons
# ✅ Matrix terminology throughout

# 3. Functionality Testing
# ✅ Sample image loading works
# ✅ Custom image upload works
# ✅ Change detection analysis runs
# ✅ Results display properly
# ✅ API endpoints respond correctly
```

### **Integration Testing**
```bash
# Backend Health Check
curl http://localhost:8000/api/health

# API Documentation
open http://localhost:8000/docs

# Frontend-Backend Integration
# Use sample images through UI
# Verify network calls in browser dev tools
# Check backend logs for MCP tool execution
```

### **Build Testing**
```bash
# Frontend Production Build
cd frontend && npm run build

# Backend Container Build  
cd backend && docker build -t matrix-detector .

# Deployment Validation
python deploy.py
```

## 📁 File Structure & Key Components

### **Critical Files**
```
├── frontend/src/components/MatrixRain.tsx    # Digital rain animation
├── frontend/src/app/globals.css             # Matrix theme CSS
├── frontend/src/app/page.tsx                # Main Matrix UI
├── backend/src/change_detector/server.py    # MCP server + FastAPI
├── backend/railway.toml                     # Railway deployment
├── backend/Dockerfile                       # Container config
├── build_and_serve.py                       # Main dev script
├── run_local.py                             # Alternative runner
├── deploy.py                                # Deployment helper
└── .gitignore                              # Includes Python cache exclusions
```

### **Environment Configuration**

#### **Backend (.env)**
```bash
OPENAI_API_KEY=your_openai_api_key_here
PORT=8000
NODE_ENV=development
```

#### **Frontend (.env.local)**
```bash
# Local Development
NEXT_PUBLIC_API_URL=http://localhost:8000

# Production (Vercel Environment Variables)
# ⚠️ CRITICAL: Must include https:// protocol!
NEXT_PUBLIC_API_URL=https://your-railway-app.railway.app
```

#### **⚠️ Common Environment Variable Mistakes**
```bash
❌ image-change-detector-production.up.railway.app   # Missing https://
❌ https://my-app.railway.app/                        # Trailing slash
❌ http://my-app.railway.app                          # Wrong protocol

✅ https://image-change-detector-production.up.railway.app  # Correct!
```

## 🎯 Implementation Status

### ✅ **COMPLETED (Production Ready)**

#### **Matrix UI (100% Complete)**
- Digital rain background with authentic characters
- Complete cyberpunk color scheme (green on black)
- Glitch text effects with RGB offset animations
- Scanline effects across interface elements
- Neon glow borders and shadows
- Matrix-themed terminology throughout
- Responsive design with mobile support

#### **MCP Backend (100% Complete)**
- FastMCP server with 4 MCP tools implemented
- FastAPI REST bridge for frontend integration
- OpenCV change detection algorithm
- GPT-4 Vision AI integration
- Error handling and validation
- Production-ready logging

#### **Frontend-Backend Integration (100% Complete)**
- HTTP API client implementation
- Base64 image encoding/decoding
- Real-time processing feedback
- Error handling and loading states
- Sample image auto-loading
- Results visualization

#### **Deployment Infrastructure (100% Complete)**
- Railway backend deployment configuration
- Vercel frontend deployment setup
- Docker containerization
- Environment variable management
- Production build optimization
- SSL/HTTPS automatic setup

#### **Development Tooling (100% Complete)**
- Automated build and serve scripts
- Dependency management (UV + NPM)
- Environment validation
- Testing helpers
- Deployment validation
- Development documentation

### 🔄 **TESTING PHASE**
- **Local Development**: All scripts functional
- **UI/UX Validation**: Matrix theme complete
- **API Integration**: Frontend-backend communication tested
- **Build Process**: Production builds successful
- **Deployment**: Ready for Railway + Vercel

### 🚀 **FUTURE ENHANCEMENTS** 
- **Authentication**: User accounts with Supabase
- **Persistence**: Image storage and analysis history
- **Advanced Algorithms**: Additional change detection methods
- **Audio Features**: Text-to-speech result narration
- **Real-time**: WebSocket connections for live updates
- **Performance**: Large image optimization and caching

## 🐛 Known Issues & Solutions

### **Resolved Issues**
1. **Python Cache Files** → Added comprehensive .gitignore exclusions
2. **UV vs PIP Conflicts** → Standardized on UV package manager
3. **MCP Tool Import Issues** → Fixed PYTHONPATH configuration
4. **Matrix Rain Performance** → Optimized Canvas rendering
5. **Production Build Errors** → Fixed TypeScript types and ESLint config
6. **⚠️ CRITICAL: Vercel Environment Variable** → MUST include `https://` protocol in `NEXT_PUBLIC_API_URL`

### **Development Notes**
- **MCP Tools**: Use `@mcp.tool()` decorator for new tools
- **Matrix Styling**: Leverage existing CSS classes in globals.css
- **API Bridge**: Add REST endpoints in server.py for frontend access
- **Environment**: Always test with and without API keys
- **Performance**: Monitor Canvas animation performance on mobile

### **Deployment Troubleshooting**
- **CORS Errors**: Check Railway logs for CORS middleware configuration
- **405 Method Not Allowed**: Verify Railway start command uses correct module path
- **Connection Refused**: Ensure Railway backend health check passes at `/api/health`
- **⚠️ Malformed URLs**: Vercel `NEXT_PUBLIC_API_URL` MUST include `https://` protocol
- **Environment Variables**: Set in both Production, Preview, and Development environments

## 🤝 Development Workflow

### **Adding New Features**
1. **MCP Tools**: Add to `server.py` with `@mcp.tool()` decorator
2. **REST API**: Add FastAPI endpoint that calls MCP tool
3. **Frontend**: Create React component with Matrix styling
4. **Testing**: Test locally with `build_and_serve.py`
5. **Documentation**: Update README.md and CLAUDE.md

### **Matrix Theme Guidelines**
- **Colors**: Use Matrix green variants (#00FF00, #00AA00, #009900)
- **Fonts**: Monospace only (Courier New, Monaco, Menlo)
- **Effects**: Use matrix-glow, matrix-pulse, matrix-scanline classes
- **Language**: Cyberpunk terminology ("NEURAL NETWORK", "MATRIX", "SURVEILLANCE")

### **Git Workflow**
```bash
# Feature development
git checkout -b feature/matrix-enhancement
# Make changes
python build_and_serve.py  # Test locally
git add .
git commit -m "Add Matrix feature"
git push origin feature/matrix-enhancement
# Create pull request
```

## 📖 Documentation

### **User Documentation**
- **README.md**: Complete user guide with deployment instructions
- **MATRIX_README.md**: Matrix theme documentation and features

### **Developer Documentation**
- **CLAUDE.md**: This file - comprehensive development knowledge
- **API Docs**: Auto-generated at localhost:8000/docs
- **Code Comments**: Inline documentation in all major functions

### **Deployment Documentation**
- **Railway Setup**: Step-by-step backend deployment
- **Vercel Setup**: Frontend deployment guide
- **Environment Config**: API key and variable setup
- **Testing Guide**: Local and production testing procedures

---

**STATUS: PRODUCTION READY** 🟢
**THEME: MATRIX CYBERPUNK** 🕶️
**ARCHITECTURE: MCP + FASTAPI** 🔧
**DEPLOYMENT: RAILWAY + VERCEL** 🌐

*"Welcome to the real world, Neo."* - The Matrix Change Detection Protocol