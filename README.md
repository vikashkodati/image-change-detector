# Image Change Detector

A satellite image change detection web application built with Next.js and Python. The system analyzes before/after satellite images to detect pixel-level changes and provides AI-powered semantic analysis of what changed.

## Current Status (Slice 2 Complete)

**✅ Implemented:**
- FastMCP server with OpenCV change detection algorithms
- Next.js frontend with professional UI components
- Frontend-backend integration via HTTP calls
- High-resolution sample images (hurricane, wildfire scenarios)
- Real-time change visualization with statistics dashboard
- GPT-4 Vision integration for semantic analysis
- Interactive Q&A capability about detected changes

**❌ Future Features:**
- Supabase authentication and storage
- Text-to-speech audio responses
- Advanced pixelmatch algorithm integration
- Performance optimization for large images

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
│  │              Shadcn/ui + Tailwind CSS                      │   │
│  │        (Button, Card, Input, Label, Textarea)              │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ HTTP Calls
                                   │ (Base64 Images)
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FastMCP Server (Port 8000)                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│  │detect_image_    │  │analyze_changes_ │  │answer_question_ │    │
│  │changes          │  │with_ai          │  │about_changes    │    │
│  │(MCP Tool)       │  │(MCP Tool)       │  │(MCP Tool)       │    │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘    │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              ChangeDetector Class                          │   │
│  │         (Core OpenCV Processing Logic)                     │   │
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
- **React Hooks** - State management and lifecycle

**Backend:**
- **FastMCP** - Tool-based API framework
- **OpenCV-Python** - Computer vision processing
- **OpenAI GPT-4 Vision** - AI-powered image analysis
- **Python 3.11+** - Runtime environment
- **UV** - Package manager

**Data Flow:**
- **Base64 encoding** - Image transport format
- **HTTP communication** - Frontend-backend protocol
- **JSON responses** - Structured data exchange

## Key Architectural Decisions

### 1. FastMCP Over REST API
**Decision:** Use FastMCP framework instead of traditional Flask/FastAPI
**Rationale:** Tool-based paradigm provides better structure for AI agent integration, self-documenting APIs, and superior error handling
**Result:** Clean separation of concerns with 4 distinct MCP tools for different operations

### 2. Monorepo Structure
**Decision:** Single repository with `frontend/` and `backend/` folders
**Rationale:** Simplifies development workflow, shared documentation, and coordinated releases
**Result:** Easy maintenance of consistency between frontend/backend contracts

### 3. Base64 Image Transport
**Decision:** Encode images as base64 strings for API transport
**Rationale:** Avoids file upload complexity, works seamlessly with JSON, simpler debugging
**Result:** Larger payload size but significantly simpler implementation

### 4. OpenCV-First Approach
**Decision:** Start with OpenCV for change detection, pixelmatch later
**Rationale:** Proven algorithms better suited for satellite imagery, more control over processing
**Result:** Robust foundation with threshold-based detection (30-pixel sensitivity)

### 5. GPT-4 Vision Integration
**Decision:** Integrate AI analysis from the beginning
**Rationale:** Provides semantic understanding beyond pixel differences
**Result:** Rich natural language descriptions of detected changes

### 6. Next.js Frontend Choice
**Decision:** Modern React framework over vanilla JavaScript
**Rationale:** TypeScript support, excellent developer experience, production-ready
**Result:** Professional UI with type safety and maintainable code

## MCP Server Implementation

The FastMCP server (`backend/src/change_detector/server.py`) provides 4 core tools:

1. **`detect_image_changes`** - OpenCV-based pixel difference detection
2. **`analyze_changes_with_ai`** - GPT-4 Vision semantic analysis
3. **`answer_question_about_changes`** - Interactive Q&A about changes
4. **`health_check`** - Server monitoring

### Change Detection Algorithm
1. Convert base64 images to OpenCV format
2. Resize images to match dimensions
3. Convert to grayscale for comparison
4. Calculate absolute difference between images
5. Apply threshold (30) to create binary mask
6. Find contours of changed regions
7. Calculate statistics and generate visualization

## Development

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Backend
```bash
cd backend
uv sync
python src/change_detector/server.py
```

## Project Structure

```
cd1/
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx       # Main UI component
│   │   │   ├── layout.tsx
│   │   │   └── globals.css
│   │   ├── components/ui/     # Shadcn/ui components
│   │   └── lib/
│   ├── public/samples/        # High-resolution sample images
│   └── package.json
├── backend/
│   ├── src/change_detector/
│   │   ├── server.py          # FastMCP server implementation
│   │   └── __init__.py
│   ├── pyproject.toml         # Python dependencies
│   └── uv.lock
├── CLAUDE.md                  # Detailed project knowledge
└── README.md
```

## Sample Images

The system includes high-resolution sample images for testing:
- **Hurricane Ian** - Florida power grid imagery (NASA Black Marble, 7680x2160)
- **LA Wildfire** - Los Angeles wildfire imagery (ESA Sentinel-2, 10m resolution)

## Environment Setup

1. Create `.env` file in backend directory:
```
OPENAI_API_KEY=your_openai_api_key_here
```

2. Install dependencies:
```bash
# Backend
cd backend && uv sync

# Frontend  
cd frontend && npm install
```

3. Run both services:
```bash
# Terminal 1: Backend
cd backend && python src/change_detector/server.py

# Terminal 2: Frontend
cd frontend && npm run dev
```

Access the application at `http://localhost:3000`