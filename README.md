# Image Change Detector

A satellite image change detection web application built with Next.js and Python. The system analyzes before/after satellite images to detect pixel-level changes and provides AI-powered semantic analysis of what changed.

## 🚀 **Live Demo**

**Frontend:** [Deploy to Vercel](https://vercel.com/import/project?template=https://github.com/your-username/your-repo)
**Backend:** [Deploy to Railway](https://railway.app/template/your-template)

## Current Status (Slice 2 Complete)

**✅ Implemented:**
- FastMCP server with OpenCV change detection algorithms
- Next.js frontend with professional UI components
- Frontend-backend integration via HTTP calls
- High-resolution sample images (hurricane, wildfire scenarios)
- Real-time change visualization with statistics dashboard
- GPT-4 Vision integration for semantic analysis
- Interactive Q&A capability about detected changes
- Modern Blotato-inspired dark UI theme
- MCP-powered backend with tool architecture

**❌ Future Features:**
- Supabase authentication and storage
- Text-to-speech audio responses
- Advanced pixelmatch algorithm integration
- Performance optimization for large images

## 🌐 **Deployment Guide**

### **1. Deploy Backend (Railway)**

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Add deployment configs"
   git push origin main
   ```

2. **Deploy to Railway:**
   - Go to [Railway.app](https://railway.app)
   - Click "Deploy from GitHub repo"
   - Select your repository
   - Set environment variables:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     PORT=8000
     ```
   - Railway will auto-detect the configuration from `railway.toml`

3. **Get your backend URL:**
   - After deployment, Railway will provide a URL like: `https://your-app.railway.app`

### **2. Deploy Frontend (Vercel)**

1. **Update environment variable:**
   Create `frontend/.env.local`:
   ```
   NEXT_PUBLIC_API_URL=https://your-backend-url.railway.app
   ```

2. **Deploy to Vercel:**
   - Go to [Vercel.com](https://vercel.com)
   - Import your GitHub repository
   - Set the root directory to `frontend`
   - Add environment variable:
     ```
     NEXT_PUBLIC_API_URL=https://your-backend-url.railway.app
     ```
   - Deploy!

### **3. Alternative: One-Click Deploy**

[![Deploy Backend](https://railway.app/button.svg)](https://railway.app/template/your-template)

[![Deploy Frontend](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/your-username/your-repo/tree/main/frontend)

## Architecture

### System Overview
```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                               │
├─────────────────────────────────────────────────────────────────────┤
│                     Next.js 14 Frontend                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│  │   Image Upload  │  │  Sample Images  │  │ Results Display │    │
│  │    Interface    │  │   Selection     │  │   Dashboard     │    │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘    │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │        Modern Dark UI (Blotato-inspired)                   │   │
│  │        (Shadcn/ui + Tailwind CSS)                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ HTTP Calls
                                   │ (Base64 Images)
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  FastMCP Server (Port 8000)                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│  │detect_image_    │  │analyze_changes_ │  │answer_question_ │    │
│  │changes          │  │with_ai          │  │about_changes    │    │
│  │(MCP Tool)       │  │(MCP Tool)       │  │(MCP Tool)       │    │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘    │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │           FastAPI + REST API Bridge                        │   │
│  │         (Calls MCP tool functions)                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                      ┌─────────────┴─────────────┐
                      │                           │
                      ▼                           ▼
            ┌─────────────────┐         ┌─────────────────┐
            │   OpenCV-Python │         │   OpenAI GPT-4  │
            │                 │         │      Vision     │
            │  • Grayscale    │         │                 │
            │  • Thresholding │         │  • Semantic     │
            │  • Contours     │         │    Analysis     │
            │  • Pixel Diff   │         │  • Natural Lang │
            │  • Statistics   │         │    Descriptions │
            └─────────────────┘         └─────────────────┘
```

### Technology Stack

**Frontend:**
- **Next.js 14** - React framework with TypeScript
- **Shadcn/ui** - Component library for consistent UI
- **Tailwind CSS** - Utility-first CSS framework
- **Modern Dark Theme** - Blotato-inspired design

**Backend:**
- **FastMCP** - Tool-based API framework with MCP tools
- **FastAPI** - REST API bridge for frontend integration
- **OpenCV-Python** - Computer vision processing
- **OpenAI GPT-4 Vision** - AI-powered image analysis
- **Python 3.11+** - Runtime environment
- **UV** - Package manager

**Deployment:**
- **Frontend:** Vercel (Static site generation)
- **Backend:** Railway (Container deployment)
- **Domain:** Custom domain support available

## Development

### Local Development
```bash
# Start both services
python build_and_serve.py

# Or manually:
# Backend
cd backend && uv sync && uv run python src/change_detector/server.py

# Frontend
cd frontend && npm install && npm run dev
```

### Environment Setup

1. **Backend environment** (`backend/.env`):
```
OPENAI_API_KEY=your_openai_api_key_here
```

2. **Frontend environment** (`frontend/.env.local`):
```
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

## Features

### 🔧 **MCP Tools Available**
- `detect_image_changes` - OpenCV change detection
- `analyze_changes_with_ai` - GPT-4 Vision analysis  
- `answer_question_about_changes` - Interactive Q&A
- `health_check` - Server monitoring

### 🎨 **Modern UI Features**
- Dark gradient theme inspired by Blotato.com
- Animated MCP status indicators
- Professional metrics dashboard
- Responsive design with glass-morphism effects
- AI-powered change visualization

### 📊 **Analysis Capabilities**
- Pixel-level change detection using OpenCV
- AI semantic analysis with GPT-4 Vision
- Statistical metrics and visualizations
- High-resolution satellite image support
- Real-time processing feedback

## Project Structure

```
cd1/
├── frontend/                    # Next.js frontend
│   ├── src/app/page.tsx        # Main UI component (Blotato-inspired)
│   ├── .env.example            # Environment variables template
│   └── next.config.mjs         # Production build config
├── backend/                     # FastMCP backend
│   ├── src/change_detector/
│   │   └── server.py           # MCP server with FastAPI bridge
│   ├── railway.toml            # Railway deployment config
│   ├── Dockerfile              # Container configuration
│   └── .env.example            # Environment variables template
├── build_and_serve.py          # Local development script
└── README.md                   # This file
```

## Sample Images

The system includes high-resolution sample images for testing:
- **Hurricane Ian** - Florida power grid imagery (NASA Black Marble, 7680x2160)
- **LA Wildfire** - Los Angeles wildfire imagery (ESA Sentinel-2, 10m resolution)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally with `python build_and_serve.py`
5. Deploy to staging environment
6. Submit a pull request

## License

MIT License - see LICENSE file for details