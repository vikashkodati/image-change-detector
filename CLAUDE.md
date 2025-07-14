# Change Detector Project Knowledge

## Project Overview
This is a satellite image change detection web application designed to analyze changes between before/after satellite images. The project is being built in incremental slices.

### Overall Scope & Vision
- **Purpose**: Comprehensive satellite image change detection and analysis platform
- **Target Users**: Researchers, analysts, and professionals working with satellite imagery
- **Core Features**: 
  - Upload before/after satellite images
  - Highlight pixel differences using multiple algorithms
  - AI-powered change descriptions via GPT-4 Vision
  - Interactive Q&A about detected changes
  - Text-to-speech audio responses
  - User authentication and session management
  - Image storage and retrieval

### Full Architecture Vision
- **Frontend**: Next.js 14 + shadcn/ui + Tailwind CSS
- **Backend**: FastMCP Python server
- **Storage**: Supabase (Auth + Storage + pgvector)
- **AI**: GPT-4 Vision + OpenAI TTS + ElevenLabs TTS
- **Change Detection**: pixelmatch + OpenCV-Python

## Current Implementation Status (Slice 1)

### ✅ Completed in Slice 1
1. **Basic Project Structure**: Frontend and backend directories established
2. **Frontend Foundation**: 
   - Next.js 14 with TypeScript setup
   - Shadcn/ui components integrated
   - Tailwind CSS configured
   - Basic image upload interface implemented
3. **Backend Foundation**:
   - FastMCP server framework setup
   - Core change detection algorithm implemented
   - OpenCV-based pixel difference detection
   - GPT-4 Vision integration for AI analysis
   - Interactive Q&A capability
4. **Core Change Detection**: Working OpenCV implementation with thresholding and contour detection

### ❌ Not Yet Implemented (Future Slices)
1. **Frontend-Backend Integration**: Connection between Next.js and FastMCP server
2. **Results Visualization**: Display of change masks, statistics, and AI analysis
3. **Advanced Change Detection**: pixelmatch algorithm integration
4. **Authentication System**: Supabase auth integration
5. **Data Persistence**: Image storage and session management
6. **Audio Features**: Text-to-speech integration (OpenAI TTS + ElevenLabs)
7. **Enhanced UI**: Advanced results display and interaction features
8. **Performance Optimization**: Large image handling and processing optimization

## Project Structure
```
cd1/
├── backend/
│   ├── src/change_detector/
│   │   ├── __init__.py
│   │   └── server.py          # Main FastMCP server implementation
│   ├── main.py                # Entry point (basic)
│   ├── pyproject.toml         # Python dependencies
│   └── README.md
└── frontend/
    ├── src/
    │   ├── app/
    │   │   ├── page.tsx       # Main UI component
    │   │   ├── layout.tsx
    │   │   └── globals.css
    │   ├── components/ui/     # Shadcn/ui components
    │   └── lib/
    └── package.json
```

## Backend Implementation

### Dependencies (pyproject.toml)
- `fastmcp>=0.1.0` - MCP server framework
- `opencv-python>=4.8.0` - Computer vision processing
- `pillow>=10.0.0` - Image handling
- `numpy>=1.24.0` - Numerical operations
- `openai>=1.0.0` - GPT-4 Vision API
- `python-dotenv>=1.0.0` - Environment variables
- `uvicorn>=0.24.0` - ASGI server
- `fastapi>=0.104.0` - Web framework
- `rasterio>=1.3.0` - Geospatial data handling

### Core Classes and Functions

#### ChangeDetector Class (server.py:27-109)
- **Purpose**: Core change detection logic using OpenCV
- **Key Methods**:
  - `detect_changes()`: Main change detection algorithm
  - `_bytes_to_cv_image()`: Convert bytes to OpenCV format
  - `_cv_image_to_base64()`: Convert OpenCV image to base64

#### MCP Tools (server.py:111-295)
1. **detect_image_changes()**: 
   - Compares two base64 images
   - Returns change percentage, pixel counts, change mask
   
2. **analyze_changes_with_ai()**:
   - Uses GPT-4 Vision for semantic analysis
   - Provides natural language description of changes
   
3. **answer_question_about_changes()**:
   - Interactive Q&A about detected changes
   - Context-aware responses using previous analysis
   
4. **health_check()**: Basic server health monitoring

### Change Detection Algorithm
1. Convert base64 images to OpenCV format
2. Resize images to match dimensions
3. Convert to grayscale for comparison
4. Calculate absolute difference between images
5. Apply threshold (30) to create binary mask
6. Find contours of changed regions
7. Calculate statistics and generate visualization

## Frontend Implementation

### Technology Stack
- **Framework**: Next.js 14 with TypeScript
- **Styling**: Tailwind CSS
- **UI Components**: Shadcn/ui
- **State Management**: React hooks

### Main Component (page.tsx)
- **File Upload**: Handles before/after image selection
- **Image Types**: Supports standard images plus .tiff, .tif, .geotiff
- **Processing State**: Loading states during analysis
- **Results Display**: Placeholder for analysis results

### Current Status
- ✅ Image upload interface complete
- ✅ File validation and state management
- ❌ Backend integration not yet implemented
- ❌ Results visualization not yet implemented

## Environment Setup

### Backend Requirements
- Python 3.11+
- OpenAI API key in `.env` file
- UV package manager (based on uv.lock presence)

### Frontend Requirements
- Node.js with npm
- Next.js 14
- Tailwind CSS configured

## Development Commands

### Backend
```bash
cd backend
# Install dependencies
uv sync

# Run server
python -m src.change_detector.server
# or
python src/change_detector/server.py
```

### Frontend
```bash
cd frontend
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build
```

## API Integration
The frontend needs to connect to the FastMCP server running on `localhost:8000` to:
1. Send base64-encoded images to `detect_image_changes`
2. Get AI analysis via `analyze_changes_with_ai`
3. Support interactive Q&A with `answer_question_about_changes`

## Next Steps for Slice 2
1. **Primary Goal**: Connect frontend to backend
2. **Key Tasks**:
   - Implement API calls from Next.js to FastMCP server
   - Add results visualization components
   - Handle image encoding/decoding between frontend and backend
   - Error handling for network requests
   - Loading states and user feedback

## Development Workflow
- **Slice-based Development**: Features implemented incrementally
- **Current Phase**: Foundation complete, integration phase next
- **Testing Strategy**: Manual testing for now, automated testing in future slices

## Technical Decisions Made
1. **FastMCP over traditional REST API**: Chosen for MCP ecosystem compatibility
2. **OpenCV for initial change detection**: Proven algorithm, good performance
3. **Base64 image encoding**: Simplifies data transfer between frontend/backend
4. **Shadcn/ui components**: Consistent, accessible UI framework
5. **Threshold value of 30**: Empirically determined for satellite image differences

## Known Limitations (Slice 1)
- No real-time processing feedback
- Single-threaded image processing
- No image format validation beyond file extensions
- Hardcoded threshold values
- No error recovery mechanisms

## Git Status
- Current branch: `main`
- Recent commits focused on frontend setup and basic structure
- Backend directory is untracked (needs to be committed)
- Ready for integration phase development