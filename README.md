# 🕶️ Matrix Change Detector

**OpenAI Agent-powered satellite image analysis with cyberpunk Matrix interface**

[![Deploy Backend](https://railway.app/button.svg)](https://railway.app/template/fastmcp-image-detector)
[![Deploy Frontend](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/your-username/your-repo/tree/main/frontend)

An intelligent satellite image change detection web application featuring an **OpenAI Agent that orchestrates FastMCP tools** for comprehensive analysis. The system uses **computer vision, AI vision analysis, and significance assessment** to detect and explain changes in satellite imagery, all wrapped in a **Matrix movie-inspired cyberpunk interface**.

## 🧠 Core Problem & Solution

**Problem**: Traditional satellite image analysis requires manual interpretation and lacks intelligent context understanding.

**Solution**: AI Agent orchestration system that:
- 🔍 **Detects** pixel-level changes using OpenCV computer vision
- 👁️ **Analyzes** semantic meaning with GPT-4 Vision
- 📊 **Assesses** significance and urgency levels automatically
- 🤖 **Synthesizes** comprehensive insights through intelligent agent reasoning

## ✨ Features

### 🎬 **Matrix-Themed Experience**
- 🌧️ **Digital Rain Background** - Authentic falling katakana characters
- 🟢 **Cyberpunk Interface** - Green-on-black terminal aesthetic
- ⚡ **Glitch Effects** - Title animations with RGB offset
- 📺 **Scanlines** - Moving lines across interface elements  
- 💚 **Neon Glow** - Green borders and shadows throughout

### 🤖 **AI Agent Intelligence**
- **OpenAI Agent Orchestration** - GPT-4 powered tool selection and reasoning
- **3-Stage Analysis Pipeline** - Detection → Vision Analysis → Significance Assessment
- **Multi-Tool Coordination** - Intelligent workflow execution
- **Significance Classification** - HIGH/MEDIUM/LOW urgency levels with recommendations
- **Comprehensive Reporting** - Synthesized insights from multiple AI systems

### 🔧 **MCP-Powered Backend**
- **FastMCP Architecture** - Modern tool-based server implementation
- **OpenCV Change Detection** - Pixel-level computer vision analysis
- **GPT-4 Vision Integration** - AI semantic understanding of changes
- **REST API Bridge** - HTTP endpoints for frontend integration
- **Real-time Processing** - Live feedback during analysis

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

1. **⚠️ CRITICAL: Set API URL Environment Variable**
   **In Vercel dashboard** → Your project → Settings → Environment Variables:
   - **Name:** `NEXT_PUBLIC_API_URL`
   - **Value:** `https://your-actual-railway-url.railway.app`
   - **⚠️ MUST include `https://` protocol** (common error!)
   - **⚠️ NO trailing slash**
   - **Environments:** ✅ Production ✅ Preview ✅ Development (check all)

   **Example correct values:**
   ```
   ✅ https://image-change-detector-production.up.railway.app
   ✅ https://my-app-production.railway.app
   ```

   **Common mistakes to avoid:**
   ```
   ❌ image-change-detector-production.up.railway.app  (missing https://)
   ❌ https://my-app.railway.app/                       (trailing slash)
   ❌ http://my-app.railway.app                         (wrong protocol)
   ```

2. **Deploy to Vercel:**
   - Visit [Vercel.com](https://vercel.com)
   - Import GitHub repository
   - Set root directory: `frontend`
   - Add environment variable (step 1 above)
   - Deploy!

### **Automated Deployment**
```bash
# Check build status and get deployment instructions
python deploy.py
```

## 🏗️ Architecture

### **AI Agent Architecture**
```
┌─────────────────────────────────────────────────────────────────────┐
│                    MATRIX USER INTERFACE                            │
├─────────────────────────────────────────────────────────────────────┤
│       Next.js 14 + Matrix Theme + Agent Configuration              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│  │ Digital Rain    │  │ Agent Control   │  │ Results Display │    │
│  │ Background      │  │ Panel           │  │ & Analytics     │    │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘    │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │    Agent Mode Selection + Query Configuration              │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ HTTP REST API
                                   │ (/api/agent-analyze)
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    OPENAI AGENT ORCHESTRATION                      │
├─────────────────────────────────────────────────────────────────────┤
│                      GPT-4 Agent Controller                        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  INTELLIGENT TOOL SELECTION & WORKFLOW EXECUTION          │   │
│  │                                                             │   │
│  │  Step 1: detect_image_changes (OpenCV)                    │   │
│  │  Step 2: analyze_images_with_gpt4_vision (AI Analysis)    │   │
│  │  Step 3: assess_change_significance (Assessment)          │   │
│  │  Step 4: Synthesize comprehensive report                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                     ┌─────────────┼─────────────┐
                     │             │             │
                     ▼             ▼             ▼
        ┌─────────────────┐ ┌─────────────┐ ┌─────────────────┐
        │   MCP TOOL 1    │ │ MCP TOOL 2  │ │   MCP TOOL 3    │
        │detect_image_    │ │analyze_     │ │assess_change_   │
        │changes          │ │images_with_ │ │significance     │
        │                 │ │gpt4_vision  │ │                 │
        │• OpenCV Proc.   │ │• GPT-4 Vision│ │• Classification │
        │• Pixel Diff     │ │• Semantic    │ │• Urgency Levels │
        │• Statistics     │ │  Analysis    │ │• Recommendations│
        │• Change Mask    │ │• Context     │ │• Pattern Recog. │
        └─────────────────┘ └─────────────┘ └─────────────────┘
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

#### 🤖 **Backend AI Stack**
```
Python 3.12 + OpenAI Agent + FastMCP
├── OpenAI Agent (GPT-4 tool orchestration)
├── FastMCP (MCP Protocol implementation)
├── FastAPI (REST API bridge)
├── OpenCV (Computer vision processing)  
├── GPT-4 Vision (AI semantic analysis)
├── NumPy (Numerical operations)
├── Uvicorn (ASGI server)
└── Tool Coordination (Agent workflow management)
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

# Test agent analysis endpoint
curl -X POST http://localhost:8000/api/agent-analyze \
  -H "Content-Type: application/json" \
  -d '{"user_query": "Analyze changes", "before_image_base64": "...", "after_image_base64": "..."}'

# Use the Matrix interface at http://localhost:3000
```

### **Matrix UI Testing**
- ✅ Digital rain animation should be visible
- ✅ Green cyberpunk color scheme throughout
- ✅ Glitch effect on main title
- ✅ Scanline animations on cards
- ✅ Glow effects on borders and buttons
- ✅ Monospace fonts (Courier New)
- ✅ Matrix-themed terminology ("NEURAL NETWORK", "MATRIX")

### **Agent Functionality Testing**
1. **Agent Mode**: Select "AI Agent Orchestrated" (recommended)
2. **Query Config**: Customize agent analysis instructions
3. **Sample Data**: Click sample images to auto-load
4. **Upload**: Try custom before/after images
5. **Analysis**: Click "EXECUTE AI AGENT ANALYSIS"
6. **Results**: Verify comprehensive agent analysis with:
   - Change detection metrics
   - GPT-4 Vision analysis
   - Significance assessment (HIGH/MEDIUM/LOW)
   - Tool execution details
7. **API**: Check backend logs for agent orchestration and MCP tool calls

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
├── .env.example                       # Environment variable template
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
- **AI Agent System**: OpenAI Agent orchestrating FastMCP tools
- **Intelligent Analysis**: 3-stage pipeline with significance assessment
- **Matrix UI**: Full cyberpunk theme with agent control panel
- **MCP Backend**: FastMCP server with 3 specialized tools
- **REST API**: Agent endpoints + legacy direct tool access
- **Sample Data**: Hurricane Ian + LA Wildfire imagery
- **Deployment**: Railway + Vercel production setup
- **Local Development**: Automated build and serve scripts
- **Comprehensive Analytics**: Multi-tool coordination with synthesis

### 🔄 **Testing Phase**
- **Agent Integration**: OpenAI Agent tool orchestration validation
- **UI/UX**: Matrix theme and agent control panel validation
- **Multi-Tool Workflow**: End-to-end agent analysis pipeline
- **Deployment**: Production environment with agent capabilities
- **Performance**: Agent response times and token optimization

### 🚀 **Future Enhancements**
- **Enhanced Agent Reasoning**: Advanced prompt engineering and tool chaining
- **Custom Agent Tools**: Domain-specific MCP tools for different imagery types
- **Multi-Agent Workflows**: Parallel analysis with specialist agents
- **Authentication**: User accounts and session management
- **Storage**: Analysis history and comparison database
- **Real-time**: WebSocket connections for live agent updates
- **Agent Memory**: Persistent context across analysis sessions

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

POWERED BY: OpenAI Agent Orchestration • GPT-4 Vision AI
           MCP Protocol v2.0 • FastAPI Matrix Interface
           OpenCV Neural Networks • Intelligent Tool Coordination
```

**[AI AGENT: ONLINE] • [MCP TOOLS: ACTIVE] • [SIGNIFICANCE ASSESSMENT: READY]**