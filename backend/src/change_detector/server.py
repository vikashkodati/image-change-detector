#!/usr/bin/env python3

import os
import asyncio
import base64
import json
from io import BytesIO
from typing import Dict, Any, Optional, List

import cv2
import numpy as np
from PIL import Image
from fastmcp import FastMCP
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client for both GPT-4 Vision analysis and Agent orchestration
openai_client = None
if os.getenv('OPENAI_API_KEY'):
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    print("✅ OpenAI client initialized for Agent orchestration and GPT-4 Vision analysis")
else:
    print("⚠️ OpenAI API key not found - Agent and GPT-4 Vision analysis will be unavailable")

# Initialize FastMCP server for tools with error handling
try:
    mcp = FastMCP("Satellite Change Detection Agent Tools")
    print("✅ FastMCP server initialized")
    
    # Get the FastAPI app from FastMCP but enhance it
    app = mcp.get_fastapi_app()
    print("✅ FastAPI app created from FastMCP")
    
except Exception as e:
    print(f"❌ FastMCP initialization failed: {e}")
    print("🔄 Falling back to pure FastAPI")
    
    # Fallback to pure FastAPI if FastMCP fails
    from fastapi import FastAPI
    app = FastAPI()
    mcp = None

# Enhance the FastAPI app configuration
app.title = "AI Agent-Powered Satellite Change Detector"
app.description = "OpenAI Agent orchestrating MCP tools for intelligent change detection"
app.version = "2.0.0"

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
    processing_mode: Optional[str] = "agent_orchestrated"
    user_query: Optional[str] = "Analyze these satellite images for changes"

class AnalysisRequest(BaseModel):
    before_image_base64: str
    after_image_base64: str
    change_results: Dict[str, Any]

class AgentRequest(BaseModel):
    before_image_base64: str
    after_image_base64: str
    user_query: str = "Analyze these satellite images for changes and provide detailed insights"

class ChangeDetector:
    """OpenCV-based change detection for MCP tools"""
    
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
            "min_contour_area": self.min_contour_area,
            "analysis_method": "opencv_computer_vision"
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

# MCP Tools for the Agent
# These will be created as standalone functions that can be called directly if MCP fails

async def detect_image_changes(before_image_base64: str, after_image_base64: str) -> Dict[str, Any]:
    """
    Detect changes between two satellite images using OpenCV computer vision.
    
    Args:
        before_image_base64: Base64 encoded image data for the "before" image
        after_image_base64: Base64 encoded image data for the "after" image
    
    Returns:
        Dictionary containing detailed change detection results and statistics
    """
    try:
        # Decode images
        before_image = decode_base64_image(before_image_base64)
        after_image = decode_base64_image(after_image_base64)
        
        # Detect changes
        results = detector.detect_changes(before_image, after_image)
        
        return {
            "success": True,
            "results": results,
            "tool_used": "opencv_change_detection"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "tool_used": "opencv_change_detection"
        }

async def analyze_images_with_gpt4_vision(before_image_base64: str, after_image_base64: str, change_context: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze satellite images using GPT-4 Vision for detailed explanations and insights.
    
    Args:
        before_image_base64: Base64 encoded image data for the "before" image  
        after_image_base64: Base64 encoded image data for the "after" image
        change_context: Optional context from change detection analysis
    
    Returns:
        Dictionary containing AI analysis of the images and changes
    """
    try:
        if not openai_client:
            return {
                "success": False,
                "error": "OpenAI API key not configured - GPT-4 Vision analysis unavailable"
            }
        
        # Parse change context if provided
        change_results = {}
        if change_context:
            try:
                change_results = json.loads(change_context) if isinstance(change_context, str) else change_context
            except:
                change_results = {"context": change_context}
        
        analysis = await analyze_changes_with_gpt4_vision(
            before_image_base64,
            after_image_base64, 
            change_results
        )
        
        if analysis:
            return {
                "success": True,
                "analysis": analysis,
                "model_used": "gpt-4-vision-preview",
                "tool_used": "gpt4_vision_analysis"
            }
        else:
            return {
                "success": False,
                "error": "GPT-4 Vision analysis failed",
                "tool_used": "gpt4_vision_analysis"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "tool_used": "gpt4_vision_analysis"
        }

async def assess_change_significance(change_percentage: float, contours_count: int, image_context: Optional[str] = None) -> Dict[str, Any]:
    """
    Assess the significance and potential impact of detected changes.
    
    Args:
        change_percentage: Percentage of image area that changed
        contours_count: Number of distinct change regions detected
        image_context: Optional context about the images being analyzed
    
    Returns:
        Dictionary containing significance assessment and recommendations
    """
    try:
        # Determine significance level
        if change_percentage > 15:
            significance = "HIGH"
            urgency = "IMMEDIATE_ATTENTION"
        elif change_percentage > 5:
            significance = "MEDIUM"
            urgency = "MONITOR_CLOSELY"
        elif change_percentage > 1:
            significance = "LOW"
            urgency = "ROUTINE_MONITORING"
        else:
            significance = "MINIMAL"
            urgency = "ARCHIVE"
        
        # Determine change pattern
        if contours_count > 20:
            pattern = "SCATTERED_MULTIPLE_CHANGES"
        elif contours_count > 5:
            pattern = "MODERATE_DISTRIBUTED_CHANGES"
        elif contours_count > 1:
            pattern = "FEW_LOCALIZED_CHANGES"
        else:
            pattern = "SINGLE_CHANGE_AREA"
        
        return {
            "success": True,
            "assessment": {
                "significance_level": significance,
                "urgency": urgency,
                "change_pattern": pattern,
                "change_percentage": change_percentage,
                "regions_count": contours_count,
                "recommendation": f"Based on {change_percentage:.2f}% change across {contours_count} regions, this requires {urgency.lower().replace('_', ' ')}."
            },
            "tool_used": "change_significance_assessment"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "tool_used": "change_significance_assessment"
        }

# Register MCP tools if FastMCP is available
if mcp:
    try:
        mcp.tool()(detect_image_changes)
        mcp.tool()(analyze_images_with_gpt4_vision)
        mcp.tool()(assess_change_significance)
        print("✅ MCP tools registered successfully")
    except Exception as e:
        print(f"⚠️ MCP tool registration failed: {e}")

# OpenAI Agent Integration
class SatelliteChangeDetectionAgent:
    """OpenAI Agent that orchestrates change detection using MCP tools"""
    
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "detect_image_changes",
                    "description": "Detect changes between two satellite images using computer vision",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "before_image_base64": {
                                "type": "string",
                                "description": "Base64 encoded before image"
                            },
                            "after_image_base64": {
                                "type": "string", 
                                "description": "Base64 encoded after image"
                            }
                        },
                        "required": ["before_image_base64", "after_image_base64"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_images_with_gpt4_vision",
                    "description": "Analyze satellite images using GPT-4 Vision for detailed insights",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "before_image_base64": {
                                "type": "string",
                                "description": "Base64 encoded before image"
                            },
                            "after_image_base64": {
                                "type": "string",
                                "description": "Base64 encoded after image"
                            },
                            "change_context": {
                                "type": "string",
                                "description": "Optional context from change detection analysis"
                            }
                        },
                        "required": ["before_image_base64", "after_image_base64"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "assess_change_significance",
                    "description": "Assess the significance and impact of detected changes",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "change_percentage": {
                                "type": "number",
                                "description": "Percentage of image area that changed"
                            },
                            "contours_count": {
                                "type": "integer",
                                "description": "Number of distinct change regions"
                            },
                            "image_context": {
                                "type": "string",
                                "description": "Optional context about the images"
                            }
                        },
                        "required": ["change_percentage", "contours_count"]
                    }
                }
            }
        ]
    
    async def analyze_satellite_images(self, before_image_base64: str, after_image_base64: str, user_query: str) -> Dict[str, Any]:
        """Main agent method to orchestrate satellite image analysis"""
        
        try:
            # System prompt for the agent
            system_prompt = """You are an expert satellite image analysis agent with access to specialized tools for change detection and assessment.

IMPORTANT: The user has already provided two satellite images (before and after) that are ready for analysis. Do NOT ask for images or any additional information. Start analysis immediately using your tools.

Your role is to:
1. Use computer vision tools to detect pixel-level changes
2. Use AI vision analysis for semantic understanding 
3. Assess the significance and implications of changes
4. Provide comprehensive, actionable insights

MANDATORY WORKFLOW - Execute immediately without asking questions:
1. FIRST: Call detect_image_changes tool to analyze pixel-level differences
2. SECOND: Call analyze_images_with_gpt4_vision tool for semantic analysis
3. THIRD: Call assess_change_significance tool based on detection results
4. FOURTH: Provide comprehensive summary combining all tool results

The satellite images are already available to your tools. Begin analysis immediately with detect_image_changes."""

            # Initial conversation with the agent
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Execute satellite image change analysis now. User request: {user_query}\n\nThe before and after images are ready. Start with detect_image_changes tool immediately."}
            ]
            
            # Start the agent conversation
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto",
                    max_tokens=500
                )
            )
            
            # Process the agent's response and tool calls
            assistant_message = response.choices[0].message
            tool_results = []
            
            if assistant_message.tool_calls:
                # Execute tool calls
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    # Add image data to tool arguments for tools that need them
                    if tool_name in ["detect_image_changes", "analyze_images_with_gpt4_vision"]:
                        tool_args["before_image_base64"] = before_image_base64
                        tool_args["after_image_base64"] = after_image_base64
                    
                    # Execute the MCP tool
                    if tool_name == "detect_image_changes":
                        result = await detect_image_changes(**tool_args)
                    elif tool_name == "analyze_images_with_gpt4_vision":
                        result = await analyze_images_with_gpt4_vision(**tool_args)
                    elif tool_name == "assess_change_significance":
                        result = await assess_change_significance(**tool_args)
                    else:
                        result = {"error": f"Unknown tool: {tool_name}"}
                    
                    tool_results.append({
                        "tool_call_id": tool_call.id,
                        "tool_name": tool_name,
                        "result": result
                    })
                    
                    # Add tool result to conversation (clean large data to avoid token limits)
                    clean_result = dict(result)
                    # Remove base64 image data from conversation to avoid token overflow
                    if "results" in clean_result and isinstance(clean_result["results"], dict):
                        if "change_mask_base64" in clean_result["results"]:
                            clean_result["results"]["change_mask_base64"] = "[IMAGE_DATA_REMOVED]"
                    
                    messages.append({
                        "role": "assistant", 
                        "content": assistant_message.content,
                        "tool_calls": assistant_message.tool_calls
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(clean_result)
                    })
                
                # Get final response from agent after tool execution
                final_response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.chat.completions.create(
                        model="gpt-4",
                        messages=messages,
                        max_tokens=500
                    )
                )
                
                return {
                    "success": True,
                    "agent_analysis": final_response.choices[0].message.content,
                    "tool_results": tool_results,
                    "tools_used": [tr["tool_name"] for tr in tool_results],
                    "orchestration_method": "openai_agent_with_mcp_tools"
                }
            
            else:
                # Agent responded without tool calls
                return {
                    "success": True,
                    "agent_analysis": assistant_message.content,
                    "tool_results": [],
                    "tools_used": [],
                    "orchestration_method": "openai_agent_direct_response"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "orchestration_method": "openai_agent_with_mcp_tools"
            }

# Initialize agent with error handling
agent = None
try:
    if openai_client:
        agent = SatelliteChangeDetectionAgent(openai_client)
        print("✅ OpenAI Agent initialized with MCP tools")
    else:
        print("⚠️ OpenAI Agent not initialized - API key not available")
except Exception as e:
    print(f"❌ OpenAI Agent initialization failed: {e}")
    print("🔄 Server will continue without agent capabilities")
    agent = None

# FastAPI endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "AI Agent-Powered Satellite Change Detection API", "status": "operational"}

@app.get("/api/health")
async def health_endpoint():
    """Health check endpoint for Railway"""
    try:
        # Test basic functionality
        test_array = np.array([[1, 2], [3, 4]])
        cv2_available = hasattr(cv2, 'cvtColor')
        
        return {
            "status": "healthy",
            "service": "AI Agent-Powered Satellite Change Detector",
            "version": "2.0.0",
            "opencv_available": cv2_available,
            "numpy_available": True,
            "openai_available": openai_client is not None,
            "agent_available": agent is not None,
            "mcp_tools_available": True
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
        "message": "AI Agent-Powered Satellite Change Detection API is operational",
        "timestamp": str(asyncio.get_event_loop().time()),
        "endpoints_available": [
            "/",
            "/api/test",
            "/api/health",
            "/api/agent-analyze",  # New agent endpoint
            "/api/detect-changes",  # Legacy endpoint
            "/api/analyze-changes"  # Legacy endpoint
        ],
        "capabilities": {
            "agent_orchestration": agent is not None,
            "opencv_detection": True,
            "gpt4_vision_analysis": openai_client is not None,
            "mcp_tools": True
        }
    }

@app.post("/api/agent-analyze")
async def agent_analyze_endpoint(request: AgentRequest):
    """Main agent-orchestrated analysis endpoint"""
    try:
        if not agent:
            return {
                "success": False,
                "error": "OpenAI Agent not available - API key may be missing or agent initialization failed",
                "fallback_suggestion": "Use /api/detect-changes for direct analysis",
                "orchestration_method": "agent_unavailable"
            }
        
        # Validate OpenAI client is still available
        if not openai_client:
            return {
                "success": False,
                "error": "OpenAI client not available - API key configuration issue",
                "orchestration_method": "openai_client_unavailable"
            }
        
        result = await agent.analyze_satellite_images(
            request.before_image_base64,
            request.after_image_base64,
            request.user_query
        )
        
        return result
        
    except Exception as e:
        print(f"Agent analysis error: {e}")
        return {
            "success": False,
            "error": f"Agent analysis failed: {str(e)}",
            "fallback_suggestion": "Use /api/detect-changes for direct analysis",
            "orchestration_method": "agent_error"
        }

@app.post("/api/detect-changes")
async def detect_changes_endpoint(request: ImagePair):
    """Legacy endpoint - direct change detection without agent"""
    try:
        # Decode images
        before_image = decode_base64_image(request.before_image_base64)
        after_image = decode_base64_image(request.after_image_base64)
        
        # Detect changes
        results = detector.detect_changes(before_image, after_image)
        
        return {
            "success": True,
            "results": results,
            "method": "direct_opencv_detection",
            "processing_mode": request.processing_mode
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "method": "direct_opencv_detection"
        }

@app.post("/api/analyze-changes")
async def analyze_changes_endpoint(request: AnalysisRequest):
    """Legacy endpoint - direct GPT-4 Vision analysis without agent"""
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
                "model_used": "gpt-4-vision-preview",
                "method": "direct_gpt4_vision"
            }
        else:
            return {
                "success": False,
                "error": "GPT-4 Vision analysis failed",
                "method": "direct_gpt4_vision"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "method": "direct_gpt4_vision"
        }

if __name__ == "__main__":
    print("🚀 Starting AI Agent-Powered Satellite Change Detection Server...")
    print(f"🔧 Python path: {os.environ.get('PYTHONPATH', 'Not set')}")
    print(f"🔑 OpenAI API key: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not set'}")
    print("🤖 Using OpenAI Agent orchestrating FastMCP tools")
    
    # Test basic imports with detailed error reporting
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
    
    # Check system status
    print("\n🔍 System Status Check:")
    print(f"📦 FastMCP available: {mcp is not None}")
    print(f"🤖 OpenAI client available: {openai_client is not None}")
    print(f"🎯 Agent available: {agent is not None}")
    print(f"🔧 App object: {type(app).__name__}")
    
    # Test health endpoint before starting server
    try:
        import asyncio
        
        async def test_health():
            try:
                # Test basic functionality
                test_array = np.array([[1, 2], [3, 4]])
                cv2_available = hasattr(cv2, 'cvtColor')
                print(f"✅ Pre-startup health check passed")
                print(f"   - NumPy test: {test_array.shape}")
                print(f"   - OpenCV available: {cv2_available}")
                return True
            except Exception as e:
                print(f"❌ Pre-startup health check failed: {e}")
                return False
        
        health_ok = asyncio.run(test_health())
        if not health_ok:
            print("⚠️ Health check failed, but attempting to start server anyway...")
    
    except Exception as e:
        print(f"⚠️ Health check error: {e}")
    
    # Run the server
    import uvicorn
    
    # Get port from environment variable for deployment platforms
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0"  # Bind to all interfaces for production
    
    print(f"\n🌐 Starting server on {host}:{port}")
    print("📡 Health check endpoint: /api/health")
    print("🧪 Test endpoint: /api/test")
    print("🤖 Agent analysis endpoint: /api/agent-analyze")
    print("🔄 Legacy endpoints: /api/detect-changes, /api/analyze-changes")
    
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        import traceback
        traceback.print_exc()
        raise