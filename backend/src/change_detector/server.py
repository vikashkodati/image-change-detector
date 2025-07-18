#!/usr/bin/env python3

import os
import asyncio
import base64
from io import BytesIO
from typing import Dict, Any, Optional
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from fastmcp import FastMCP
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client with error handling
try:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  Warning: OPENAI_API_KEY not set. AI analysis features will be disabled.")
        client = None
    else:
        client = OpenAI(api_key=api_key)
        print("✅ OpenAI client initialized successfully")
except Exception as e:
    print(f"⚠️  Warning: Failed to initialize OpenAI client: {e}")
    print("   AI analysis features will be disabled.")
    client = None

# Initialize FastMCP server
mcp = FastMCP("Image Change Detector Server")

# Pydantic models for REST API
class DetectChangesRequest(BaseModel):
    before_image_base64: str
    after_image_base64: str

class AnalyzeChangesRequest(BaseModel):
    before_image_base64: str
    after_image_base64: str
    change_results: Dict[str, Any]

# Create a proper FastAPI app that uses MCP tools
def create_fastapi_with_mcp():
    """Create FastAPI app with REST endpoints that call MCP tools"""
    from fastapi import FastAPI
    
    # Create FastAPI app (this will work properly)
    app = FastAPI(title="Image Change Detector API (MCP-powered)")
    
    # Add CORS middleware - allow all origins for public deployment
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins for public access
        allow_credentials=False,  # Set to False when allowing all origins
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    
    # REST API endpoints that call MCP tool functions
    @app.post("/api/detect-changes")
    async def api_detect_changes(request: DetectChangesRequest):
        """REST API endpoint that calls the same logic as MCP detect_image_changes tool"""
        try:
            # Decode base64 images
            before_bytes = base64.b64decode(request.before_image_base64)
            after_bytes = base64.b64decode(request.after_image_base64)
            
            # Perform change detection using the detector directly
            results = await detector.detect_changes(before_bytes, after_bytes)
            
            # Check if the detection had errors
            if "error" in results:
                return {
                    "success": False,
                    "error": results["error"]
                }
            
            return {
                "success": True,
                "results": results
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    @app.post("/api/analyze-changes")
    async def api_analyze_changes(request: AnalyzeChangesRequest):
        """REST API endpoint that calls the same logic as MCP analyze_changes_with_ai tool"""
        try:
            # Prepare messages for GPT-4 Vision
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert in satellite image analysis. Analyze the before and after images to describe what changed."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Please analyze these two satellite images and describe the changes. Change statistics: {request.change_results.get('change_percentage', 0):.2f}% of pixels changed."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{request.before_image_base64}"
                            }
                        },
                        {
                            "type": "image_url", 
                            "image_url": {
                                "url": f"data:image/png;base64,{request.after_image_base64}"
                            }
                        }
                    ]
                }
            ]
            
            # Call GPT-4 Vision
            if client is None:
                return {
                    "success": False,
                    "error": "OpenAI client not available. Please set OPENAI_API_KEY environment variable."
                }
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=500
            )
            
            analysis = response.choices[0].message.content
            
            return {
                "success": True,
                "analysis": analysis,
                "model_used": "gpt-4o"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"AI analysis failed: {str(e)}"
            }
    
    @app.get("/api/health")
    async def api_health():
        """REST API health check that provides the same info as MCP health_check tool"""
        return {
            "status": "healthy",
            "service": "Image Change Detector Server (MCP-powered)",
            "version": "0.1.0",
            "mcp_tools_available": True
        }
    
    # Add MCP tools documentation endpoint
    @app.get("/mcp/tools")
    async def list_mcp_tools():
        """List available MCP tools"""
        return {
            "tools": [
                {
                    "name": "detect_image_changes",
                    "description": "Detect changes between two satellite images using OpenCV"
                },
                {
                    "name": "analyze_changes_with_ai", 
                    "description": "Use GPT-4 Vision to analyze and describe changes"
                },
                {
                    "name": "answer_question_about_changes",
                    "description": "Answer specific questions about changes in images"
                },
                {
                    "name": "health_check",
                    "description": "Server health check"
                }
            ],
            "server": "FastMCP-powered Image Change Detector",
            "mcp_available": True
        }
    
    return app

class ChangeDetector:
    """Core change detection logic"""
    
    def __init__(self):
        self.temp_dir = Path("/tmp/change_detector")
        self.temp_dir.mkdir(exist_ok=True)
    
    async def detect_changes(self, before_image: bytes, after_image: bytes) -> Dict[str, Any]:
        """
        Detect changes between two images using OpenCV
        """
        try:
            # Convert bytes to OpenCV images
            before_cv = self._bytes_to_cv_image(before_image)
            after_cv = self._bytes_to_cv_image(after_image)
            
            # Ensure images are the same size
            if before_cv.shape != after_cv.shape:
                # Resize after image to match before image
                after_cv = cv2.resize(after_cv, (before_cv.shape[1], before_cv.shape[0]))
            
            # Convert to grayscale for comparison
            before_gray = cv2.cvtColor(before_cv, cv2.COLOR_BGR2GRAY)
            after_gray = cv2.cvtColor(after_cv, cv2.COLOR_BGR2GRAY)
            
            # Calculate absolute difference
            diff = cv2.absdiff(before_gray, after_gray)
            
            # Apply threshold to get binary mask
            _, threshold = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            
            # Find contours (changed regions)
            contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create change mask visualization
            change_mask = np.zeros_like(before_cv)
            cv2.drawContours(change_mask, contours, -1, (0, 255, 0), thickness=cv2.FILLED)
            
            # Calculate change statistics
            total_pixels = before_gray.shape[0] * before_gray.shape[1]
            changed_pixels = cv2.countNonZero(threshold)
            change_percentage = (changed_pixels / total_pixels) * 100
            
            # Convert change mask to base64 for frontend
            change_mask_base64 = self._cv_image_to_base64(change_mask)
            
            return {
                "change_percentage": round(change_percentage, 2),
                "changed_pixels": int(changed_pixels),
                "total_pixels": int(total_pixels),
                "change_mask_base64": change_mask_base64,
                "contours_count": len(contours)
            }
            
        except Exception as e:
            return {
                "error": f"Change detection failed: {str(e)}",
                "change_percentage": 0,
                "changed_pixels": 0,
                "total_pixels": 0,
                "change_mask_base64": "",
                "contours_count": 0
            }
    
    def _bytes_to_cv_image(self, image_bytes: bytes) -> np.ndarray:
        """Convert bytes to OpenCV image"""
        # Try to handle different image formats
        try:
            # First try as regular image
            image = Image.open(BytesIO(image_bytes))
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        except Exception:
            # If that fails, try as numpy array directly
            nparr = np.frombuffer(image_bytes, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    def _cv_image_to_base64(self, cv_image: np.ndarray) -> str:
        """Convert OpenCV image to base64 string"""
        _, buffer = cv2.imencode('.png', cv_image)
        return base64.b64encode(buffer).decode('utf-8')

# Initialize change detector
detector = ChangeDetector()

# Create the FastAPI app at module level for Railway
app = None

def get_or_create_app():
    """Get or create the FastAPI app"""
    global app
    if app is None:
        app = create_fastapi_with_mcp()
    return app

# Make sure app is available at module level for Railway
try:
    print("🚀 Initializing FastAPI app for Railway...")
    app = get_or_create_app()
    print("✅ FastAPI app initialized successfully for Railway")
except Exception as e:
    print(f"❌ Failed to initialize FastAPI app: {e}")
    raise

@mcp.tool()
async def detect_image_changes(before_image_base64: str, after_image_base64: str) -> Dict[str, Any]:
    """
    Detect changes between two satellite images
    
    Args:
        before_image_base64: Base64 encoded before image
        after_image_base64: Base64 encoded after image
        
    Returns:
        Dictionary containing change detection results
    """
    try:
        # Decode base64 images
        before_bytes = base64.b64decode(before_image_base64)
        after_bytes = base64.b64decode(after_image_base64)
        
        # Perform change detection
        results = await detector.detect_changes(before_bytes, after_bytes)
        
        return {
            "success": True,
            "results": results
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
async def analyze_changes_with_ai(
    before_image_base64: str, 
    after_image_base64: str,
    change_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Use GPT-4 Vision to analyze and describe the changes
    
    Args:
        before_image_base64: Base64 encoded before image
        after_image_base64: Base64 encoded after image  
        change_results: Results from change detection
        
    Returns:
        Dictionary containing AI analysis
    """
    try:
        # Prepare messages for GPT-4 Vision
        messages = [
            {
                "role": "system",
                "content": "You are an expert in satellite image analysis. Analyze the before and after images to describe what changed."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Please analyze these two satellite images and describe the changes. Change statistics: {change_results.get('change_percentage', 0):.2f}% of pixels changed."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{before_image_base64}"
                        }
                    },
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url": f"data:image/png;base64,{after_image_base64}"
                        }
                    }
                ]
            }
        ]
        
        # Call GPT-4 Vision
        if client is None:
            return {
                "success": False,
                "error": "OpenAI client not available. Please set OPENAI_API_KEY environment variable."
            }
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=500
        )
        
        analysis = response.choices[0].message.content
        
        return {
            "success": True,
            "analysis": analysis,
            "model_used": "gpt-4o"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"AI analysis failed: {str(e)}"
        }

@mcp.tool()
async def answer_question_about_changes(
    question: str,
    before_image_base64: str,
    after_image_base64: str,
    previous_analysis: str = ""
) -> Dict[str, Any]:
    """
    Answer specific questions about the changes in the images
    
    Args:
        question: User's question about the changes
        before_image_base64: Base64 encoded before image
        after_image_base64: Base64 encoded after image
        previous_analysis: Previous AI analysis (optional)
        
    Returns:
        Dictionary containing the answer
    """
    try:
        # Prepare messages for GPT-4 Vision
        context = f"Previous analysis: {previous_analysis}" if previous_analysis else ""
        
        messages = [
            {
                "role": "system",
                "content": "You are an expert in satellite image analysis. Answer questions about changes between before and after images."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{context}\n\nQuestion: {question}\n\nPlease analyze these images and answer the question."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{before_image_base64}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{after_image_base64}"
                        }
                    }
                ]
            }
        ]
        
        # Call GPT-4 Vision
        if client is None:
            return {
                "success": False,
                "error": "OpenAI client not available. Please set OPENAI_API_KEY environment variable."
            }
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=300
        )
        
        answer = response.choices[0].message.content
        
        return {
            "success": True,
            "answer": answer,
            "question": question,
            "model_used": "gpt-4o"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Question answering failed: {str(e)}"
        }

# Basic health check
@mcp.tool()
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Image Change Detector Server",
        "version": "0.1.0"
    }

if __name__ == "__main__":
    print("🚀 Starting Matrix Change Detector Server...")
    print(f"🔧 Python path: {os.environ.get('PYTHONPATH', 'Not set')}")
    print(f"🔑 OpenAI API key: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not set'}")
    
    # Use the global app
    print("📱 Using FastAPI app with MCP tools...")
    print("✅ FastAPI app created successfully")
    
    # Run the server
    import uvicorn
    
    # Get port from environment variable for deployment platforms
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0"  # Bind to all interfaces for production
    
    print(f"🌐 Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
    
    # Note: The MCP tools are available and can be used by MCP clients
    # The FastAPI endpoints provide REST access to the same functionality