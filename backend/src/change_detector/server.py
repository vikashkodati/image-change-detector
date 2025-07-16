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
import rasterio
from fastmcp import FastMCP
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

# Create a hybrid FastAPI + FastMCP server
def create_app():
    """Create FastAPI app with REST endpoints and integrate with FastMCP"""
    from fastapi import FastAPI
    
    # Create separate FastAPI app for REST endpoints
    app = FastAPI(title="Image Change Detector API")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    
    # REST API endpoints for frontend integration
    @app.post("/api/detect-changes")
    async def api_detect_changes(request: DetectChangesRequest):
        """REST API endpoint that calls the change detection logic"""
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
        """REST API endpoint for AI analysis"""
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
        """REST API health check"""
        return {
            "status": "healthy",
            "service": "Image Change Detector Server",
            "version": "0.1.0"
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
    import uvicorn
    
    # Create the FastAPI app
    app = create_app()
    
    # Run the server
    uvicorn.run(app, host="127.0.0.1", port=8000)