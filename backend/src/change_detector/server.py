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
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# Hybrid AI imports - lazy loading for Railway deployment
CLIP_AVAILABLE = False
CLIP_IMPORT_ERROR = None

def check_clip_availability():
    """Check if CLIP is available without importing heavy dependencies"""
    global CLIP_AVAILABLE, CLIP_IMPORT_ERROR
    if CLIP_AVAILABLE:
        return True
    
    try:
        import torch
        import clip
        CLIP_AVAILABLE = True
        print("✅ CLIP dependencies verified and available")
        return True
    except ImportError as e:
        CLIP_AVAILABLE = False
        CLIP_IMPORT_ERROR = str(e)
        print(f"⚠️  CLIP dependencies not available: {e}")
        print("   Will fall back to OpenCV-only detection")
        return False
    except Exception as e:
        CLIP_AVAILABLE = False
        CLIP_IMPORT_ERROR = str(e)
        print(f"⚠️  CLIP check failed: {e}")
        return False

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
    # More explicit CORS configuration for Railway
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Go back to wildcard - Railway issue
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],
        allow_headers=["*"],
        allow_origin_regex=None,
    )
    
    # Add explicit OPTIONS handling for preflight requests
    @app.options("/api/{path:path}")
    async def options_handler(path: str):
        return {
            "message": "OK",
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "*"
            }
        }
    
    # REST API endpoints that call MCP tool functions
    @app.post("/api/detect-changes")
    async def api_detect_changes(request: DetectChangesRequest):
        """REST API endpoint that calls the same logic as MCP detect_image_changes tool"""
        try:
            # Decode base64 images
            before_bytes = base64.b64decode(request.before_image_base64)
            after_bytes = base64.b64decode(request.after_image_base64)
            
            # Perform hybrid change detection (OpenCV + CLIP)
            results = await detector.detect_changes_hybrid(before_bytes, after_bytes)
            
            # Check if the detection had errors
            if "error" in results:
                response_data = {
                    "success": False,
                    "error": results["error"],
                    "method": "hybrid_detection"
                }
            else:
                response_data = {
                    "success": True,
                    "results": results,
                    "method": "hybrid_detection",
                    "enhanced_features": {
                        "semantic_analysis": results.get("semantic_results", {}).get("available", False),
                        "change_classification": results.get("classification_results", {}).get("available", False),
                        "threat_assessment": results.get("final_assessment", {}).get("threat_level", "UNKNOWN")
                    }
                }
            response = JSONResponse(content=response_data)
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "*"
            return response
            
        except Exception as e:
            response_data = {
                "success": False,
                "error": str(e)
            }
            response = JSONResponse(content=response_data)
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "*"
            return response
    
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
                    "name": "detect_changes_hybrid_mcp",
                    "description": "Hybrid change detection using OpenCV + CLIP semantic analysis"
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
            "server": "FastMCP-powered Image Change Detector (Hybrid AI)",
            "mcp_available": True,
            "hybrid_features": {
                "semantic_analysis": semantic_analyzer is not None,
                "change_classification": check_clip_availability(),
                "threat_assessment": True
            }
        }
    
    # New Hybrid AI endpoints
    @app.post("/api/semantic-analysis")
    async def api_semantic_analysis(request: DetectChangesRequest):
        """REST API endpoint for CLIP semantic similarity analysis"""
        if not semantic_analyzer:
            response_data = {
                "success": False,
                "error": "Semantic analyzer not available",
                "available": False
            }
        else:
            try:
                before_bytes = base64.b64decode(request.before_image_base64)
                after_bytes = base64.b64decode(request.after_image_base64)
                
                results = await semantic_analyzer.analyze_semantic_similarity(before_bytes, after_bytes)
                
                response_data = {
                    "success": True,
                    "results": results,
                    "method": "clip_semantic_analysis"
                }
                
            except Exception as e:
                response_data = {
                    "success": False,
                    "error": str(e),
                    "method": "clip_semantic_analysis"
                }
        
        response = JSONResponse(content=response_data)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        return response
    
    @app.post("/api/classify-change")
    async def api_classify_change(request: DetectChangesRequest):
        """REST API endpoint for CLIP change type classification"""
        if not semantic_analyzer:
            response_data = {
                "success": False,
                "error": "Semantic analyzer not available",
                "available": False
            }
        else:
            try:
                before_bytes = base64.b64decode(request.before_image_base64)
                after_bytes = base64.b64decode(request.after_image_base64)
                
                results = await semantic_analyzer.classify_change_type(before_bytes, after_bytes)
                
                response_data = {
                    "success": True,
                    "results": results,
                    "method": "clip_change_classification"
                }
                
            except Exception as e:
                response_data = {
                    "success": False,
                    "error": str(e),
                    "method": "clip_change_classification"
                }
        
        response = JSONResponse(content=response_data)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        return response
    
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
    
    async def detect_changes_hybrid(self, before_image: bytes, after_image: bytes) -> Dict[str, Any]:
        """
        Hybrid change detection: OpenCV + CLIP semantic analysis
        Stage 1: Fast OpenCV pixel detection
        Stage 2: CLIP semantic validation and classification
        """
        try:
            print("🔍 Starting hybrid change detection...")
            
            # Stage 1: OpenCV pixel-level detection
            print("   Stage 1: OpenCV pixel detection...")
            opencv_results = await self.detect_changes(before_image, after_image)
            
            # Check if OpenCV detected any changes
            if "error" in opencv_results:
                return {
                    "hybrid_available": True,
                    "method": "opencv_only",
                    "opencv_results": opencv_results,
                    "semantic_results": {"available": False, "error": "OpenCV failed"},
                    "final_assessment": {
                        "has_meaningful_change": False,
                        "confidence": 0.0,
                        "reasoning": f"OpenCV detection failed: {opencv_results['error']}"
                    }
                }
            
            change_percentage = opencv_results.get("change_percentage", 0)
            
            # Early exit if no pixel changes detected
            if change_percentage < 0.5:
                print("   No significant pixel changes detected - skipping CLIP analysis")
                return {
                    "hybrid_available": True,
                    "method": "opencv_early_exit",
                    "opencv_results": opencv_results,
                    "semantic_results": {"available": False, "reason": "No pixel changes to analyze"},
                    "final_assessment": {
                        "has_meaningful_change": False,
                        "confidence": 0.95,
                        "reasoning": "No significant pixel-level changes detected"
                    }
                }
            
            # Stage 2: CLIP semantic analysis (only if pixel changes found)
            print("   Stage 2: CLIP semantic validation...")
            semantic_results = {}
            classification_results = {}
            
            if semantic_analyzer:
                # Semantic similarity analysis (will handle lazy loading internally)
                semantic_results = await semantic_analyzer.analyze_semantic_similarity(
                    before_image, after_image
                )
                
                # Change type classification (will handle lazy loading internally)
                classification_results = await semantic_analyzer.classify_change_type(
                    before_image, after_image
                )
            else:
                semantic_results = {"available": False, "error": "Semantic analyzer not initialized"}
                classification_results = {"available": False, "error": "Semantic analyzer not initialized"}
            
            # Combine results and make final assessment
            final_assessment = self._assess_hybrid_results(
                opencv_results, semantic_results, classification_results
            )
            
            print(f"   Hybrid assessment: {final_assessment['has_meaningful_change']} (confidence: {final_assessment['confidence']})")
            
            return {
                "hybrid_available": True,
                "method": "full_hybrid" if semantic_results.get("available") else "opencv_only",
                "opencv_results": opencv_results,
                "semantic_results": semantic_results,
                "classification_results": classification_results,
                "final_assessment": final_assessment,
                "processing_time": {
                    "opencv_stage": "~100ms",
                    "clip_stage": "~200ms" if semantic_results.get("available") else "skipped"
                }
            }
            
        except Exception as e:
            return {
                "hybrid_available": True,
                "error": f"Hybrid detection failed: {str(e)}",
                "method": "error",
                "final_assessment": {
                    "has_meaningful_change": False,
                    "confidence": 0.0,
                    "reasoning": f"System error: {str(e)}"
                }
            }
    
    def _assess_hybrid_results(self, opencv_results: Dict, semantic_results: Dict, 
                              classification_results: Dict) -> Dict[str, Any]:
        """Combine OpenCV and CLIP results to make final assessment"""
        
        change_percentage = opencv_results.get("change_percentage", 0)
        
        # If CLIP is not available, fall back to OpenCV only
        if not semantic_results.get("available", False):
            # Conservative threshold for OpenCV-only decisions
            has_change = change_percentage > 2.0  # Higher threshold without semantic filtering
            confidence = 0.7 if has_change else 0.8  # Lower confidence without CLIP
            
            return {
                "has_meaningful_change": has_change,
                "confidence": confidence,
                "reasoning": f"OpenCV-only: {change_percentage:.2f}% pixel changes (CLIP unavailable)",
                "change_type": "unknown",
                "threat_level": "MEDIUM" if has_change else "LOW"
            }
        
        # Get CLIP semantic analysis
        semantic_similarity = semantic_results.get("semantic_similarity", 1.0)
        is_meaningful = semantic_results.get("is_meaningful_change", False)
        change_confidence = semantic_results.get("change_confidence", 0.0)
        
        # Get classification results
        most_likely_change = classification_results.get("most_likely", "unknown")
        max_confidence = classification_results.get("max_confidence", 0.0)
        
        # Decision logic: Combine pixel changes + semantic analysis
        if change_percentage > 10.0:
            # High pixel changes - trust CLIP semantic validation
            if is_meaningful and change_confidence > 0.1:
                result = {
                    "has_meaningful_change": True,
                    "confidence": min(0.95, 0.8 + change_confidence),
                    "reasoning": f"High pixel changes ({change_percentage:.2f}%) + semantic validation (similarity: {semantic_similarity:.3f})",
                    "change_type": most_likely_change,
                    "threat_level": self._determine_threat_level(most_likely_change, change_confidence)
                }
            else:
                result = {
                    "has_meaningful_change": False,
                    "confidence": 0.85,
                    "reasoning": f"High pixel changes ({change_percentage:.2f}%) but semantically similar (similarity: {semantic_similarity:.3f}) - likely noise",
                    "change_type": "noise_or_artifacts",
                    "threat_level": "LOW"
                }
        elif change_percentage > 2.0:
            # Medium pixel changes - CLIP is decisive
            if is_meaningful:
                result = {
                    "has_meaningful_change": True,
                    "confidence": 0.7 + change_confidence * 0.2,
                    "reasoning": f"Moderate pixel changes ({change_percentage:.2f}%) with semantic differences (similarity: {semantic_similarity:.3f})",
                    "change_type": most_likely_change,
                    "threat_level": self._determine_threat_level(most_likely_change, change_confidence)
                }
            else:
                result = {
                    "has_meaningful_change": False,
                    "confidence": 0.80,
                    "reasoning": f"Moderate pixel changes ({change_percentage:.2f}%) but semantically very similar (similarity: {semantic_similarity:.3f})",
                    "change_type": "minor_variations",
                    "threat_level": "LOW"
                }
        else:
            # Low pixel changes - probably noise
            result = {
                "has_meaningful_change": False,
                "confidence": 0.90,
                "reasoning": f"Low pixel changes ({change_percentage:.2f}%) - insufficient for meaningful change",
                "change_type": "no_change",
                "threat_level": "NONE"
            }
        
        return result
    
    def _determine_threat_level(self, change_type: str, confidence: float) -> str:
        """Determine threat level based on change type and confidence"""
        disaster_keywords = ["destruction", "damage", "fire", "flood", "disaster"]
        infrastructure_keywords = ["construction", "development", "urban", "road"]
        environmental_keywords = ["vegetation", "deforestation", "drought"]
        
        if any(keyword in change_type.lower() for keyword in disaster_keywords):
            return "HIGH" if confidence > 0.3 else "MEDIUM"
        elif any(keyword in change_type.lower() for keyword in infrastructure_keywords):
            return "MEDIUM" if confidence > 0.2 else "LOW"
        elif any(keyword in change_type.lower() for keyword in environmental_keywords):
            return "MEDIUM" if confidence > 0.25 else "LOW"
        else:
            return "LOW"
    
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

class CLIPSemanticAnalyzer:
    """CLIP-based semantic change analysis for hybrid detection"""
    
    def __init__(self):
        self.model = None
        self.preprocess = None
        self.device = None
        self.category_embeddings = {}
        self.initialization_attempted = False
        self.initialization_error = None
        
        self.change_categories = [
            "building destruction", "building construction", "building damage",
            "vegetation loss", "vegetation growth", "deforestation",
            "flooding", "drought", "water level change",
            "fire damage", "burn scars", "smoke",
            "road construction", "infrastructure development",
            "urban expansion", "demolition",
            "seasonal change", "snow cover", "cloud shadows",
            "agricultural change", "crop harvesting", "farming activity"
        ]
        
        # Don't initialize during startup - do it lazily when first needed
        print("📱 CLIPSemanticAnalyzer created (lazy initialization)")
    
    def _initialize_model(self):
        """Initialize CLIP model and preprocessing (lazy loading)"""
        if self.initialization_attempted:
            return self.model is not None
        
        self.initialization_attempted = True
        
        try:
            # Check if CLIP is available first
            if not check_clip_availability():
                self.initialization_error = CLIP_IMPORT_ERROR or "CLIP not available"
                return False
            
            # Now import the heavy dependencies
            import torch
            import clip
            
            print("🧠 Initializing CLIP model for semantic analysis...")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Use smaller ViT-B/32 model for better performance
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            self.model.eval()
            
            # Precompute text embeddings for change categories
            self.category_embeddings = self._precompute_category_embeddings()
            print(f"✅ CLIP model loaded on {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize CLIP model: {e}")
            self.initialization_error = str(e)
            self.model = None
            return False
    
    def _precompute_category_embeddings(self):
        """Precompute embeddings for change categories"""
        if not self.model:
            return {}
        
        try:
            category_texts = [f"satellite image showing {category}" for category in self.change_categories]
            text_tokens = clip.tokenize(category_texts).to(self.device)
            
            with torch.no_grad():
                text_embeddings = self.model.encode_text(text_tokens)
                text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            
            return {
                category: embedding for category, embedding 
                in zip(self.change_categories, text_embeddings)
            }
        except Exception as e:
            print(f"⚠️  Failed to precompute category embeddings: {e}")
            return {}
    
    def _preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        """Preprocess image for CLIP"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            
            # Apply CLIP preprocessing
            preprocessed = self.preprocess(image).unsqueeze(0).to(self.device)
            return preprocessed
            
        except Exception as e:
            print(f"❌ Image preprocessing failed: {e}")
            return None
    
    async def analyze_semantic_similarity(self, before_image: bytes, after_image: bytes) -> Dict[str, Any]:
        """Analyze semantic similarity between two images"""
        # Lazy initialization on first use
        if not self._initialize_model():
            return {
                "available": False,
                "error": f"CLIP model not available: {self.initialization_error or 'Unknown error'}"
            }
        
        try:
            # Preprocess images
            before_tensor = self._preprocess_image(before_image)
            after_tensor = self._preprocess_image(after_image)
            
            if before_tensor is None or after_tensor is None:
                return {"error": "Image preprocessing failed"}
            
            # Generate image embeddings
            with torch.no_grad():
                before_embedding = self.model.encode_image(before_tensor)
                after_embedding = self.model.encode_image(after_tensor)
                
                # Normalize embeddings
                before_embedding = before_embedding / before_embedding.norm(dim=-1, keepdim=True)
                after_embedding = after_embedding / after_embedding.norm(dim=-1, keepdim=True)
                
                # Calculate cosine similarity
                similarity = torch.cosine_similarity(before_embedding, after_embedding).item()
            
            # Determine if change is meaningful
            # Similarity > 0.95 = very similar (likely noise)
            # Similarity < 0.85 = significant semantic change
            is_meaningful = similarity < 0.90
            change_confidence = 1 - similarity if similarity < 0.90 else 0.0
            
            return {
                "available": True,
                "semantic_similarity": round(similarity, 4),
                "is_meaningful_change": is_meaningful,
                "change_confidence": round(change_confidence, 4),
                "interpretation": self._interpret_similarity(similarity)
            }
            
        except Exception as e:
            return {
                "available": True,
                "error": f"Semantic analysis failed: {str(e)}"
            }
    
    async def classify_change_type(self, before_image: bytes, after_image: bytes) -> Dict[str, Any]:
        """Classify the type of change using CLIP"""
        # Lazy initialization on first use
        if not self._initialize_model():
            return {
                "available": False,
                "error": f"CLIP classification not available: {self.initialization_error or 'Unknown error'}"
            }
        
        try:
            # Generate embedding for the "after" image (shows the change result)
            after_tensor = self._preprocess_image(after_image)
            if after_tensor is None:
                return {"error": "Image preprocessing failed"}
            
            with torch.no_grad():
                image_embedding = self.model.encode_image(after_tensor)
                image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
                
                # Calculate similarities with all categories
                similarities = {}
                for category, text_embedding in self.category_embeddings.items():
                    similarity = torch.cosine_similarity(
                        image_embedding, 
                        text_embedding.unsqueeze(0)
                    ).item()
                    similarities[category] = similarity
                
                # Get top 3 most likely change types
                top_categories = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:3]
                
                return {
                    "available": True,
                    "top_categories": [
                        {
                            "category": category,
                            "confidence": round(similarity, 4),
                            "likelihood": "high" if similarity > 0.25 else "medium" if similarity > 0.15 else "low"
                        }
                        for category, similarity in top_categories
                    ],
                    "most_likely": top_categories[0][0] if top_categories else "unknown",
                    "max_confidence": round(top_categories[0][1], 4) if top_categories else 0.0
                }
                
        except Exception as e:
            return {
                "available": True,
                "error": f"Change classification failed: {str(e)}"
            }
    
    def _interpret_similarity(self, similarity: float) -> str:
        """Interpret semantic similarity score"""
        if similarity > 0.95:
            return "Virtually identical - likely noise or compression artifacts"
        elif similarity > 0.90:
            return "Very similar - minor lighting or seasonal changes"
        elif similarity > 0.80:
            return "Moderately similar - some changes detected"
        elif similarity > 0.70:
            return "Somewhat different - notable changes present"
        elif similarity > 0.60:
            return "Quite different - significant changes detected"
        else:
            return "Very different - major changes or completely different scenes"

# Initialize semantic analyzer (lazy loading)
semantic_analyzer = CLIPSemanticAnalyzer()  # Always create, will lazy-load CLIP when needed

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

@mcp.tool()
async def detect_changes_hybrid_mcp(before_image_base64: str, after_image_base64: str) -> Dict[str, Any]:
    """
    Hybrid change detection using OpenCV + CLIP semantic analysis
    
    Args:
        before_image_base64: Base64 encoded before image
        after_image_base64: Base64 encoded after image
        
    Returns:
        Dictionary containing hybrid analysis results with pixel detection + semantic validation
    """
    try:
        # Decode base64 images
        before_bytes = base64.b64decode(before_image_base64)
        after_bytes = base64.b64decode(after_image_base64)
        
        # Perform hybrid detection
        results = await detector.detect_changes_hybrid(before_bytes, after_bytes)
        
        return {
            "success": True,
            "results": results,
            "method": "hybrid_detection",
            "capabilities": {
                "opencv_available": True,
                "clip_available": check_clip_availability(),
                "semantic_analysis": semantic_analyzer is not None
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "method": "hybrid_detection"
        }

# Basic health check
@mcp.tool()
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Image Change Detector Server (Hybrid AI)",
        "version": "0.1.0",
        "capabilities": {
            "opencv": True,
            "clip": check_clip_availability(),  # Check lazily
            "semantic_analysis": semantic_analyzer is not None,
            "hybrid_detection": True
        }
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