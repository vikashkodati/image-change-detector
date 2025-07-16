# 🕶️ Matrix Change Detector

**AI-powered satellite image analysis with a cyberpunk Matrix-themed interface**

[![Deploy Backend](https://railway.app/button.svg)](https://railway.app/template/fastmcp-image-detector)
[![Deploy Frontend](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/your-username/your-repo/tree/main/frontend)

A satellite image change detection web application built with **Next.js** and **FastMCP**. The system analyzes before/after satellite images to detect pixel-level changes and provides AI-powered semantic analysis with a **Matrix movie-inspired cyberpunk interface**.

## ✨ Features

### 🎬 **Matrix-Themed Experience**
- 🌧️ **Digital Rain Background** - Authentic falling katakana characters
- 🟢 **Cyberpunk Interface** - Green-on-black terminal aesthetic
- ⚡ **Glitch Effects** - Title animations with RGB offset
- 📺 **Scanlines** - Moving lines across interface elements  
- 💚 **Neon Glow** - Green borders and shadows throughout

### 🧠 **MCP-Powered Analysis**
- **OpenCV Change Detection** - Pixel-level difference analysis
- **GPT-4 Vision Integration** - AI semantic analysis of changes
- **FastMCP Architecture** - Tool-based backend with REST API bridge
- **Real-time Processing** - Live feedback during analysis
- **High-resolution Support** - Handles large satellite imagery

### 📡 **Sample Data**
- **Hurricane Ian** - Florida power grid analysis (NASA Black Marble 7680x2160)
- **LA Wildfires** - Los Angeles wildfire monitoring (Sentinel-2 10m resolution)

## 🚀 Quick Start

### **Method 1: Automated Setup (Recommended)**
```bash
# Clone and navigate
git clone <your-repo-url>
cd cd1

# Quick start (installs deps + starts both services)
python build_and_serve.py
```

### **Method 2: Alternative Local Runner**
```bash
# Alternative startup script
python run_local.py
```

### **Method 3: Manual Setup**
```bash
# Backend (Terminal 1)
cd backend
uv sync
uv run python -m change_detector.server

# Frontend (Terminal 2)  
cd frontend
npm install
npm run dev
```

### **Access Your Matrix**
- 🖥️ **Frontend**: http://localhost:3000 (or 3001 if busy)
- 🔧 **Backend**: http://localhost:8000
- 📖 **API Docs**: http://localhost:8000/docs

## 🔧 Environment Setup

### **Backend Configuration**
Create `backend/.env`:
```env
# Required: OpenAI API key for GPT-4 Vision analysis
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Local development settings
NODE_ENV=development
PORT=8000
```

### **Frontend Configuration** 
Create `frontend/.env.local`:
```env
# API endpoint (auto-detected for local development)
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 🌐 Production Deployment

### **Backend: Railway**

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Deploy Matrix Change Detector"
   git push origin main
   ```

2. **Deploy to Railway:**
   - Visit [Railway.app](https://railway.app) 
   - "Deploy from GitHub repo" → Select your repository
   - Set environment variables:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     PORT=8000
     ```
   - Railway auto-detects config from `railway.toml`

3. **Get backend URL:** Railway provides URL like `https://your-app.railway.app`

### **Frontend: Vercel**

1. **Update API URL:**
   Add to Vercel environment variables:
   ```
   NEXT_PUBLIC_API_URL=https://your-backend-url.railway.app
   ```

2. **Deploy to Vercel:**
   - Visit [Vercel.com](https://vercel.com)
   - Import GitHub repository
   - Set root directory: `frontend`
   - Add environment variable
   - Deploy!

### **Automated Deployment**
```bash
# Check build status and get deployment instructions
python deploy.py
```

## 🏗️ Architecture

### **System Overview**
```
┌─────────────────────────────────────────────────────────────────────┐
│                    MATRIX USER INTERFACE                            │
├─────────────────────────────────────────────────────────────────────┤
│                 Next.js 14 + Matrix Theme                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│  │ Digital Rain    │  │ Cyberpunk UI    │  │ Glitch Effects  │    │
│  │ Background      │  │ Components      │  │ & Animations    │    │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘    │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │        Matrix Canvas + Shadcn/ui + Tailwind CSS            │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ HTTP REST API
                                   │ (Base64 Images)
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│              FastMCP Server + FastAPI Bridge                       │
├─────────────────────────────────────────────────────────────────────┤
│                       MCP TOOLS AVAILABLE:                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│  │detect_image_    │  │analyze_changes_ │  │answer_question_ │    │
│  │changes          │  │with_ai          │  │about_changes    │    │
│  │(OpenCV)         │  │(GPT-4 Vision)   │  │(Interactive)    │    │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘    │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │         REST API Endpoints (/api/detect-changes)           │   │
│  │           (Bridge to MCP Tool Functions)                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                      ┌─────────────┴─────────────┐
                      │                           │
                      ▼                           ▼
            ┌─────────────────┐         ┌─────────────────┐
            │   OpenCV        │         │   OpenAI        │
            │   Computer      │         │   GPT-4 Vision  │
            │   Vision        │         │      API        │
            │                 │         │                 │
            │  • Grayscale    │         │  • Semantic     │
            │  • Thresholding │         │    Analysis     │
            │  • Contours     │         │  • Natural Lang │
            │  • Pixel Diff   │         │    Descriptions │
            │  • Statistics   │         │  • Interactive  │
            └─────────────────┘         └─────────────────┘
```

### **Technology Stack**

#### 🖥️ **Frontend Matrix**
```
Next.js 14 + TypeScript
├── React Components (Matrix-themed UI)
├── Canvas API (Digital rain animation)  
├── Tailwind CSS (Cyberpunk styling)
├── Shadcn/ui (Component library)
└── Matrix CSS animations (Glitch, scanlines, glow)
```

#### 🔧 **Backend Matrix**
```
Python 3.12 + FastMCP
├── FastMCP (MCP Protocol implementation)
├── FastAPI (REST API bridge)
├── OpenCV (Computer vision processing)
├── GPT-4 Vision (AI semantic analysis)
├── NumPy (Numerical operations)
└── Uvicorn (ASGI server)
```

#### 🌐 **Deployment Stack**
```
Production Deployment
├── Frontend: Vercel (Static site generation)
├── Backend: Railway (Container deployment)
├── Domain: Custom domain support
├── SSL: Automatic HTTPS
└── CDN: Global edge distribution
```

## 🧪 Testing

### **Local Testing**
```bash
# Test full application
python build_and_serve.py

# Test backend only
cd backend && uv run python -m change_detector.server

# Test frontend only  
cd frontend && npm run dev

# Test build process
python deploy.py
```

### **API Testing**
```bash
# Health check
curl http://localhost:8000/api/health

# API documentation
open http://localhost:8000/docs

# Test change detection (with sample files)
# Use the frontend interface at http://localhost:3000
```

### **Matrix UI Testing**
- ✅ Digital rain animation should be visible
- ✅ Green cyberpunk color scheme throughout
- ✅ Glitch effect on main title
- ✅ Scanline animations on cards
- ✅ Glow effects on borders and buttons
- ✅ Monospace fonts (Courier New)
- ✅ Matrix-themed terminology ("NEURAL NETWORK", "MATRIX")

### **Functionality Testing**
1. **Sample Data**: Click sample images to auto-load
2. **Upload**: Try custom before/after images 
3. **Analysis**: Click "EXECUTE MCP ANALYSIS"
4. **Results**: Verify change detection matrix displays
5. **API**: Check backend logs for MCP tool calls

## 📁 Project Structure

```
cd1/
├── frontend/                           # Next.js Matrix-themed frontend
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx               # Main Matrix UI component
│   │   │   ├── globals.css            # Matrix CSS styles
│   │   │   └── layout.tsx             # App layout
│   │   ├── components/
│   │   │   ├── MatrixRain.tsx         # Digital rain animation
│   │   │   └── ui/                    # Matrix-themed shadcn components
│   │   └── lib/utils.ts
│   ├── next.config.mjs                # Production build config
│   ├── package.json                   # Frontend dependencies
│   └── .env.example                   # Environment template
│
├── backend/                            # FastMCP backend server  
│   ├── src/change_detector/
│   │   └── server.py                  # MCP server + FastAPI bridge
│   ├── pyproject.toml                 # Python dependencies
│   ├── railway.toml                   # Railway deployment config
│   ├── Dockerfile                     # Container configuration
│   └── .env.example                   # Environment template
│
├── build_and_serve.py                 # Local development script
├── run_local.py                       # Alternative startup script  
├── deploy.py                          # Deployment helper script
├── README.md                          # This file
├── MATRIX_README.md                   # Matrix theme documentation
├── CLAUDE.md                          # Development knowledge base
└── .gitignore                         # Git exclusions (includes Python cache)
```

## 🛠️ Development

### **Adding New Features**
1. **MCP Tools**: Add new tools in `backend/src/change_detector/server.py`
2. **UI Components**: Create in `frontend/src/components/`
3. **Matrix Styling**: Use classes from `globals.css` (matrix-*, glitch, etc.)
4. **API Endpoints**: Add REST bridges in server.py

### **Matrix Theme Guidelines**
- **Colors**: Use Matrix green (`#00FF00`) variants
- **Fonts**: Monospace (Courier New, Monaco, Menlo)
- **Effects**: Leverage matrix-glow, matrix-pulse, matrix-scanline
- **Terminology**: Use cyberpunk/Matrix language ("NEURAL NETWORK", "SURVEILLANCE DATA")

### **Dependencies**
```bash
# Backend dependencies
cd backend && uv add <package-name>

# Frontend dependencies  
cd frontend && npm install <package-name>

# Update lock files
cd backend && uv lock
cd frontend && npm update
```

## 🎯 Current Status

### ✅ **Implemented (Complete)**
- **Matrix UI**: Full cyberpunk theme with digital rain
- **MCP Backend**: FastMCP server with OpenCV + GPT-4 Vision
- **REST API**: Frontend-backend integration via HTTP
- **Sample Data**: Hurricane Ian + LA Wildfire imagery
- **Deployment**: Railway + Vercel production setup
- **Local Development**: Automated build and serve scripts
- **Documentation**: Comprehensive architecture and deployment guides

### 🔄 **Testing Phase**
- **UI/UX**: Matrix theme visual validation
- **Integration**: Frontend-backend API communication
- **Deployment**: Production environment validation
- **Performance**: Large image processing optimization

### 🚀 **Future Enhancements**
- **Authentication**: User accounts and session management  
- **Storage**: Image persistence and history
- **Advanced Algorithms**: Additional change detection methods
- **Audio**: Text-to-speech result narration
- **Real-time**: WebSocket connections for live updates

## 📖 Additional Documentation

- **Matrix Theme**: See [MATRIX_README.md](./MATRIX_README.md)
- **Development Notes**: See [CLAUDE.md](./CLAUDE.md)
- **Deployment Helper**: Run `python deploy.py`

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/matrix-enhancement`
3. Test locally: `python build_and_serve.py`
4. Commit changes: `git commit -m "Add Matrix feature"`
5. Push and create pull request

## 📄 License

MIT License - see LICENSE file for details

---

```
"There is no spoon. Only data."
- The Matrix Change Detection Protocol

POWERED BY: OpenCV Neural Networks • GPT-4 Vision AI 
           MCP Protocol v2.0 • FastAPI Matrix Interface
```

**[SYSTEM STATUS: ONLINE] • [MCP TOOLS: ACTIVE] • [NEURAL NETWORK: READY]**