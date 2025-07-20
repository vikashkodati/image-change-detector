#!/usr/bin/env python3

import os
import asyncio
import base64
from io import BytesIO
from typing import Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client for GPT-4 Vision analysis
openai_client = None
if os.getenv('OPENAI_API_KEY'):
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    print("✅ OpenAI client initialized for GPT-4 Vision analysis")
else:
    print("⚠️ OpenAI API key not found - GPT-4 Vision analysis will be unavailable")

# Create FastAPI app
app = FastAPI(
    title="Satellite Image Change Detector",
    description="OpenCV + GPT-4 Vision change detection service",
    version="1.0.0"
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class ImagePair(BaseModel):
    before_image_base64: str
    after_image_base64: str
    processing_mode: Optional[str] = "opencv_only"

class AnalysisRequest(BaseModel):
    before_image_base64: str
    after_image_base64: str
    change_results: Dict[str, Any]

class ChangeDetector:
    """Simple OpenCV-based change detection"""
    
    def __init__(self):
        self.min_contour_area = 100
        self.gaussian_blur_ksize = (5, 5)
        self.dilation_kernel_size = (3, 3)
        self.threshold_value = 30
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for change detection"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, self.gaussian_blur_ksize, 0)
        
        return blurred
    
    def detect_changes(self, before_image: np.ndarray, after_image: np.ndarray) -> Dict[str, Any]:
        """Detect changes between two images using OpenCV"""
        
        # Ensure images are the same size
        h, w = before_image.shape[:2]
        after_image = cv2.resize(after_image, (w, h))
        
        # Preprocess images
        before_gray = self.preprocess_image(before_image)
        after_gray = self.preprocess_image(after_image)
        
        # Compute absolute difference
        diff = cv2.absdiff(before_gray, after_gray)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(diff, self.threshold_value, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.dilation_kernel_size)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        filtered_contours = [c for c in contours if cv2.contourArea(c) > self.min_contour_area]
        
        # Calculate statistics
        total_pixels = before_gray.size
        changed_pixels = np.sum(thresh > 0)
        change_percentage = (changed_pixels / total_pixels) * 100
        
        # Create change mask visualization
        change_mask = np.zeros_like(before_image)
        if len(change_mask.shape) == 3:
            change_mask[:, :, 1] = thresh  # Green channel for changes
        else:
            change_mask = thresh
        
        # Convert change mask to base64 for display
        _, buffer = cv2.imencode('.png', change_mask)
        change_mask_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "change_percentage": float(change_percentage),
            "changed_pixels": int(changed_pixels),
            "total_pixels": int(total_pixels),
            "contours_count": len(filtered_contours),
            "change_mask_base64": change_mask_base64,
            "threshold_used": self.threshold_value,
            "min_contour_area": self.min_contour_area
        }

# Initialize detector
detector = ChangeDetector()

def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 image string to numpy array"""
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # Decode base64
    image_data = base64.b64decode(base64_string)
    
    # Convert to PIL Image
    pil_image = Image.open(BytesIO(image_data))
    
    # Convert to RGB if needed
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # Convert to OpenCV format (BGR)
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    return cv_image

async def analyze_changes_with_gpt4_vision(before_base64: str, after_base64: str, change_results: Dict[str, Any]) -> Optional[str]:
    """Analyze changes using GPT-4 Vision"""
    if not openai_client:
        return None
    
    try:
        # Prepare the prompt with context
        change_percentage = change_results.get("change_percentage", 0)
        contours_count = change_results.get("contours_count", 0)
        
        prompt = f"""You are an expert satellite imagery analyst. Analyze these two images and provide a clear, professional explanation of what has changed.

Context from computer vision analysis:
- {change_percentage:.2f}% of the image area shows changes
- {contours_count} distinct change regions detected

Please provide:
1. A clear description of what you observe in the before and after images
2. Specific changes you can identify (infrastructure, environmental, etc.)
3. Potential significance or implications of these changes
4. Assessment of whether the detected change percentage seems accurate

Keep your analysis concise but informative, as if briefing a decision-maker."""

        # Prepare images for API
        before_data_url = f"data:image/jpeg;base64,{before_base64.split(',')[-1]}"
        after_data_url = f"data:image/jpeg;base64,{after_base64.split(',')[-1]}"
        
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: openai_client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": before_data_url}
                            },
                            {
                                "type": "image_url", 
                                "image_url": {"url": after_data_url}
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"GPT-4 Vision analysis failed: {e}")
        return None

# FastAPI endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Satellite Image Change Detector API", "status": "operational"}

@app.get("/api/health")
async def health_endpoint():
    """Health check endpoint for Railway"""
    try:
        # Test basic functionality
        test_array = np.array([[1, 2], [3, 4]])
        cv2_available = hasattr(cv2, 'cvtColor')
        
        return {
            "status": "healthy",
            "service": "Satellite Image Change Detector",
            "version": "1.0.0",
            "opencv_available": cv2_available,
            "numpy_available": True,
            "openai_available": openai_client is not None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/api/test")
async def test_endpoint():
    """Test endpoint to verify API is working"""
    return {
        "message": "Satellite Change Detection API is operational",
        "timestamp": str(asyncio.get_event_loop().time()),
        "endpoints_available": [
            "/",
            "/api/test",
            "/api/health",
            "/api/detect-changes", 
            "/api/analyze-changes"
        ],
        "capabilities": {
            "opencv_detection": True,
            "gpt4_vision_analysis": openai_client is not None
        }
    }

@app.post("/api/detect-changes")
async def detect_changes_endpoint(request: ImagePair):
    """Detect changes between two images"""
    try:
        # Decode images
        before_image = decode_base64_image(request.before_image_base64)
        after_image = decode_base64_image(request.after_image_base64)
        
        # Detect changes
        results = detector.detect_changes(before_image, after_image)
        
        return {
            "success": True,
            "results": results,
            "method": "opencv_detection",
            "processing_mode": request.processing_mode
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "method": "opencv_detection"
        }

@app.post("/api/analyze-changes")
async def analyze_changes_endpoint(request: AnalysisRequest):
    """Analyze changes using GPT-4 Vision"""
    try:
        if not openai_client:
            return {
                "success": False,
                "error": "OpenAI API key not configured - GPT-4 Vision analysis unavailable"
            }
        
        analysis = await analyze_changes_with_gpt4_vision(
            request.before_image_base64,
            request.after_image_base64,
            request.change_results
        )
        
        if analysis:
            return {
                "success": True,
                "analysis": analysis,
                "model_used": "gpt-4-vision-preview"
            }
        else:
            return {
                "success": False,
                "error": "GPT-4 Vision analysis failed"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    print("🚀 Starting Satellite Change Detection Server...")
    print(f"🔧 Python path: {os.environ.get('PYTHONPATH', 'Not set')}")
    print(f"🔑 OpenAI API key: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not set'}")
    print("📱 Using Pure FastAPI server with OpenCV + GPT-4 Vision analysis")
    
    # Test basic imports
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
    except Exception as e:
        print(f"❌ OpenCV import failed: {e}")
    
    try:
        import numpy as np
        print(f"✅ NumPy version: {np.__version__}")
    except Exception as e:
        print(f"❌ NumPy import failed: {e}")
    
    # Run the server
    import uvicorn
    
    # Get port from environment variable for deployment platforms
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0"  # Bind to all interfaces for production
    
    print(f"🌐 Starting server on {host}:{port}")
    print("📡 Health check endpoint: /api/health")
    print("🧪 Test endpoint: /api/test")
    
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        raise