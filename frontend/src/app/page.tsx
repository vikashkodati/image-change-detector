"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import MatrixRain from "@/components/MatrixRain";

// TypeScript interface for sample image data
interface SampleImage {
  id: number;
  name: string;
  before: string;
  after: string;
  description: string;
  resolution: string;
  clipEngagement: string;
}

// Type definitions for agent results
interface AgentToolResult {
  tool_call_id: string;
  tool_name: string;
  result: {
    success: boolean;
    results?: any;
    analysis?: string;
    assessment?: any;
    error?: string;
    tool_used: string;
  };
}

interface AgentResults {
  success: boolean;
  agent_analysis?: string;
  tool_results?: AgentToolResult[];
  tools_used?: string[];
  orchestration_method?: string;
  error?: string;
}

// Legacy results structure for backward compatibility
interface LegacyResults {
  success: boolean;
  method?: string;
  processing_mode?: string;
  results?: {
    change_percentage?: number;
    changed_pixels?: number;
    total_pixels?: number;
    change_mask_base64?: string;
    contours_count?: number;
    analysis_method?: string;
  };
  ai_analysis?: string;
  model_used?: string;
  error?: string;
}

  // Curated sample image sets for change detection analysis
  const sampleImages: SampleImage[] = [
    {
      id: 1,
      name: "Hurricane Ian - Power Grid Impact",
      before: "/samples/hurricane_ian_before.png",
      after: "/samples/hurricane_ian_after.png",
      description: "Dramatic nighttime satellite imagery showing power grid disruption across Florida during Hurricane Ian (September 2022)",
      resolution: "NASA Black Marble 500m",
      clipEngagement: "Power infrastructure analysis"
    },
    {
      id: 2,
      name: "Los Angeles Wildfire Smoke",
      before: "/samples/la_wildfire_current.jpg",
      after: "/samples/la_wildfire_current.jpg",
      description: "Wildfire smoke plumes and atmospheric changes captured by satellite over Los Angeles region (January 2025)",
      resolution: "Sentinel-2 10m",
      clipEngagement: "Atmospheric conditions analysis"
    },
    {
      id: 3,
      name: "Comparison Test - Same Images",
      before: "/samples/hurricane_ian_before.png",
      after: "/samples/hurricane_ian_before.png",
      description: "Control test using identical images to demonstrate detection accuracy and false positive handling",
      resolution: "NASA Black Marble 500m",
      clipEngagement: "Baseline comparison test"
    }
  ];

export default function Home() {
  const [beforeImage, setBeforeImage] = useState<File | null>(null);
  const [afterImage, setAfterImage] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState<AgentResults | LegacyResults | null>(null);
  const [selectedSample, setSelectedSample] = useState<number | null>(null);
  const [analysisMode, setAnalysisMode] = useState<string>("agent_orchestrated");
  const [userQuery, setUserQuery] = useState<string>("Analyze these satellite images for changes and provide detailed insights about what has changed, including significance and implications.");

  // API URL for both local and production
  const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

  // Convert File to base64 string
  const fileToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => {
        const result = reader.result as string;
        const base64 = result.split(',')[1];
        resolve(base64);
      };
      reader.onerror = (error) => reject(error);
    });
  };

  // Agent-orchestrated analysis (new primary method)
  const handleAgentAnalysis = async () => {
    if (!beforeImage || !afterImage) {
      alert("Please select both before and after images.");
      return;
    }

    setIsProcessing(true);
    setResults(null);

    try {
      const beforeBase64 = await fileToBase64(beforeImage);
      const afterBase64 = await fileToBase64(afterImage);

      const response = await fetch(`${API_URL}/api/agent-analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          before_image_base64: beforeBase64,
          after_image_base64: afterBase64,
          user_query: userQuery
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: AgentResults = await response.json();
      setResults(data);

    } catch (error) {
      console.error('Agent analysis error:', error);
      setResults({
        success: false,
        error: 'Failed to process images with AI agent. Please check your connection and try again.',
        orchestration_method: "agent_error"
      });
    } finally {
      setIsProcessing(false);
    }
  };

  // Legacy direct detection (backup method)
  const handleDirectDetection = async () => {
    if (!beforeImage || !afterImage) {
      alert("Please select both before and after images.");
      return;
    }

    setIsProcessing(true);
    setResults(null);

    try {
      const beforeBase64 = await fileToBase64(beforeImage);
      const afterBase64 = await fileToBase64(afterImage);

      const response = await fetch(`${API_URL}/api/detect-changes`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          before_image_base64: beforeBase64,
          after_image_base64: afterBase64,
          processing_mode: "opencv_only",
        }),
      });

      const data = await response.json();
      setResults(data);

      // If detection was successful, get GPT-4 Vision analysis
      if (data.success && data.results && !data.error) {
        try {
          console.log('Getting GPT-4 Vision analysis...');
          const analysisResponse = await fetch(`${API_URL}/api/analyze-changes`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              before_image_base64: beforeBase64,
              after_image_base64: afterBase64,
              change_results: data.results
            }),
          });

          if (analysisResponse.ok) {
            const analysisData = await analysisResponse.json();
            if (analysisData.success) {
              // Add AI analysis to results
              setResults(prevResults => ({
                ...prevResults!,
                ai_analysis: analysisData.analysis,
                model_used: analysisData.model_used
              }));
              console.log('GPT-4 Vision analysis completed');
            }
          }
        } catch (analysisError) {
          console.error('GPT-4 Vision analysis failed:', analysisError);
          // Don't show error to user - just continue without AI analysis
        }
      }
    } catch (error) {
      console.error('Error:', error);
      setResults({
        success: false,
        error: 'Failed to process images. Please check your connection and try again.'
      });
    } finally {
      setIsProcessing(false);
    }
  };

  // Main handler that delegates based on analysis mode
  const handleAnalysis = () => {
    if (analysisMode === "agent_orchestrated") {
      handleAgentAnalysis();
    } else {
      handleDirectDetection();
    }
  };

  const loadSampleImages = async (sample: SampleImage) => {
    setSelectedSample(sample.id);
    
    try {
      // Convert URLs to File objects
      const beforeResponse = await fetch(sample.before);
      const beforeBlob = await beforeResponse.blob();
      const beforeFile = new File([beforeBlob], `${sample.name}_before.png`, { type: 'image/png' });

      const afterResponse = await fetch(sample.after);
      const afterBlob = await afterResponse.blob();
      const afterFile = new File([afterBlob], `${sample.name}_after.png`, { type: 'image/png' });

      setBeforeImage(beforeFile);
      setAfterImage(afterFile);
    } catch (error) {
      console.error('Error loading sample images:', error);
    }
  };

  // Extract change detection data from agent results for display
  const getChangeDetectionData = (results: AgentResults | LegacyResults) => {
    if (!results.success) return null;

    // Check if it's agent results
    if ('tool_results' in results && results.tool_results) {
      const detectionTool = results.tool_results.find(tool => tool.tool_name === 'detect_image_changes');
      if (detectionTool?.result?.success && detectionTool.result.results) {
        return detectionTool.result.results;
      }
    }

    // Check if it's legacy results
    if ('results' in results && results.results) {
      return results.results;
    }

    return null;
  };

  // Extract significance assessment from agent results
  const getSignificanceAssessment = (results: AgentResults) => {
    if (!results.success || !results.tool_results) return null;

    const assessmentTool = results.tool_results.find(tool => tool.tool_name === 'assess_change_significance');
    if (assessmentTool?.result?.success && assessmentTool.result.assessment) {
      return assessmentTool.result.assessment;
    }

    return null;
  };

  // Extract GPT-4 Vision analysis (either from agent or legacy)
  const getVisionAnalysis = (results: AgentResults | LegacyResults) => {
    if (!results.success) return null;

    // Check if it's agent results
    if ('tool_results' in results && results.tool_results) {
      const visionTool = results.tool_results.find(tool => tool.tool_name === 'analyze_images_with_gpt4_vision');
      if (visionTool?.result?.success && visionTool.result.analysis) {
        return visionTool.result.analysis;
      }
    }

    // Check if it's legacy results
    if ('ai_analysis' in results && results.ai_analysis) {
      return results.ai_analysis;
    }

    return null;
  };

  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* Matrix Rain Background */}
      <MatrixRain />
      
      {/* Matrix-themed UI */}
      <div className="relative z-10 p-6">
        {/* Header with glitch effect */}
        <div className="text-center mb-8">
          <h1 
            className="matrix-glitch text-6xl font-bold mb-4 matrix-text"
            data-text="MATRIX CHANGE DETECTOR"
          >
            MATRIX CHANGE DETECTOR
          </h1>
          <div className="matrix-scanline">
            <p className="text-xl matrix-text opacity-80">
              🤖 AI AGENT-POWERED SATELLITE ANALYSIS SYSTEM 🤖
            </p>
          </div>
        </div>

        {/* Enhanced Status Banner */}
        <div className="matrix-card matrix-glow p-4 mb-8 rounded-lg matrix-scanline">
          <div className="flex items-center justify-center space-x-2 flex-wrap">
            <div className="matrix-pulse">
              <div className="w-3 h-3 bg-green-400 rounded-full"></div>
            </div>
            <span className="matrix-text font-mono text-lg">
              [MATRIX AGENT SYSTEM: ONLINE] • AI-ORCHESTRATED SATELLITE ANALYSIS ACTIVE
            </span>
            <div className="matrix-pulse">
              <div className="w-3 h-3 bg-green-400 rounded-full"></div>
            </div>
          </div>
          <div className="flex items-center justify-center space-x-2 mt-2 flex-wrap">
            <span className="matrix-text font-mono text-sm opacity-80">
              🤖 OpenAI Agent • 🔍 MCP Tools • 🛰️ Computer Vision • 👁️ GPT-4 Vision • 📊 Significance Assessment
            </span>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          
          {/* Left Panel: Upload Interface */}
          <Card className="matrix-card matrix-glow">
            <CardHeader className="matrix-scanline">
              <CardTitle className="matrix-text text-2xl font-mono">
                {'>>'} AI AGENT CONTROL PANEL
              </CardTitle>
              <CardDescription className="matrix-text opacity-70">
                Configure AI agent parameters and upload satellite imagery
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">

              {/* Analysis Mode Selection */}
              <div>
                <Label className="matrix-text text-lg font-mono block mb-4">
                  &gt; ANALYSIS MODE:
                </Label>
                <div className="grid grid-cols-1 gap-3">
                  <div
                    className={`p-4 rounded-lg border-2 cursor-pointer transition-all duration-300 ${
                      analysisMode === "agent_orchestrated"
                        ? 'matrix-glow border-green-400 bg-green-900/20'
                        : 'matrix-border hover:border-green-400 hover:bg-green-900/10'
                    }`}
                    onClick={() => setAnalysisMode("agent_orchestrated")}
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <h3 className="matrix-text font-mono font-bold">🤖 AI Agent Orchestrated</h3>
                        <p className="matrix-text text-sm opacity-70">
                          Advanced AI agent uses multiple tools for comprehensive analysis
                        </p>
                      </div>
                      <div className="px-2 py-1 rounded text-xs font-mono font-bold bg-green-900/50 text-green-300 border border-green-500">
                        RECOMMENDED
                      </div>
                    </div>
                  </div>
                  
                  <div
                    className={`p-4 rounded-lg border-2 cursor-pointer transition-all duration-300 ${
                      analysisMode === "direct_detection"
                        ? 'matrix-glow border-green-400 bg-green-900/20'
                        : 'matrix-border hover:border-green-400 hover:bg-green-900/10'
                    }`}
                    onClick={() => setAnalysisMode("direct_detection")}
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <h3 className="matrix-text font-mono font-bold">🔍 Direct Detection</h3>
                        <p className="matrix-text text-sm opacity-70">
                          Traditional OpenCV + GPT-4 Vision analysis (legacy mode)
                        </p>
                      </div>
                      <div className="px-2 py-1 rounded text-xs font-mono font-bold bg-blue-900/50 text-blue-300 border border-blue-500">
                        LEGACY
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Agent Query Configuration (only for agent mode) */}
              {analysisMode === "agent_orchestrated" && (
                <div>
                  <Label className="matrix-text text-lg font-mono block mb-4">
                    &gt; AGENT ANALYSIS QUERY:
                  </Label>
                  <div className="matrix-border p-3 rounded-lg bg-blue-900/10 mb-4">
                    <p className="matrix-text font-mono text-sm opacity-80">
                      🤖 <span className="text-blue-400">AGENT INSTRUCTION:</span> Customize what you want the AI agent to focus on during analysis
                    </p>
                  </div>
                  <Textarea
                    value={userQuery}
                    onChange={(e) => setUserQuery(e.target.value)}
                    placeholder="Analyze these satellite images for changes..."
                    className="matrix-border matrix-text bg-black border-green-400 min-h-[100px] resize-none"
                    rows={4}
                  />
                </div>
              )}

              {/* Sample Data Section */}
              <div>
                <Label className="matrix-text text-lg font-mono block mb-4">
                  &gt; SATELLITE IMAGERY SAMPLES:
                </Label>
                <div className="matrix-border p-3 rounded-lg bg-blue-900/10 mb-4">
                  <p className="matrix-text font-mono text-sm opacity-80">
                    🛰️ <span className="text-blue-400">CURATED DATASETS:</span> Real satellite imagery from major events with detailed change analysis
                  </p>
                </div>
                <div className="grid gap-4">
                  {sampleImages.map((sample) => (
                    <div
                      key={sample.id}
                      className={`p-4 rounded-lg border-2 cursor-pointer transition-all duration-300 ${
                        selectedSample === sample.id
                          ? 'matrix-glow border-green-400 bg-green-900/20'
                          : 'matrix-border hover:border-green-400 hover:bg-green-900/10'
                      }`}
                      onClick={() => loadSampleImages(sample)}
                    >
                      <div className="flex justify-between items-start mb-2">
                        <h3 className="matrix-text font-mono font-bold text-lg">
                          {sample.name}
                        </h3>
                        <div className="px-2 py-1 rounded text-xs font-mono font-bold bg-blue-900/50 text-blue-300 border border-blue-500">
                          🛰️ {sample.clipEngagement}
                        </div>
                      </div>
                      
                      <p className="matrix-text opacity-70 text-sm mb-2">
                        {sample.description}
                      </p>
                      
                      <p className="matrix-text opacity-50 text-xs font-mono">
                        RESOLUTION: {sample.resolution}
                      </p>
                    </div>
                  ))}
                </div>
              </div>

              {/* File Upload Section */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="before-image" className="matrix-text font-mono">
                    &gt; BEFORE IMAGE:
                  </Label>
                  <Input
                    id="before-image"
                    type="file"
                    accept="image/*"
                    onChange={(e) => setBeforeImage(e.target.files?.[0] || null)}
                    className="matrix-border matrix-text bg-black border-green-400 file:matrix-button file:border-green-400 file:text-green-400"
                  />
                  {beforeImage && (
                    <p className="matrix-text text-sm mt-2 opacity-70">
                      ✓ {beforeImage.name}
                    </p>
                  )}
                </div>

                <div>
                  <Label htmlFor="after-image" className="matrix-text font-mono">
                    &gt; AFTER IMAGE:
                  </Label>
                  <Input
                    id="after-image"
                    type="file"
                    accept="image/*"
                    onChange={(e) => setAfterImage(e.target.files?.[0] || null)}
                    className="matrix-border matrix-text bg-black border-green-400 file:matrix-button file:border-green-400 file:text-green-400"
                  />
                  {afterImage && (
                    <p className="matrix-text text-sm mt-2 opacity-70">
                      ✓ {afterImage.name}
                    </p>
                  )}
                </div>
              </div>

              {/* Image Preview Section */}
              {(beforeImage || afterImage) && (
                <div className="mb-6">
                  <Label className="matrix-text text-lg font-mono block mb-4">
                    &gt; IMAGE PREVIEW:
                  </Label>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="matrix-border rounded-lg p-4 bg-green-900/10">
                      <h3 className="matrix-text font-mono font-bold mb-2">📸 BEFORE IMAGE</h3>
                      {beforeImage ? (
                        <div className="matrix-border rounded overflow-hidden">
                          <img
                            src={URL.createObjectURL(beforeImage)}
                            alt="Before"
                            className="w-full h-48 object-cover"
                          />
                        </div>
                      ) : (
                        <div className="matrix-border rounded bg-black/50 h-48 flex items-center justify-center">
                          <span className="matrix-text opacity-50">No image selected</span>
                        </div>
                      )}
                    </div>
                    <div className="matrix-border rounded-lg p-4 bg-green-900/10">
                      <h3 className="matrix-text font-mono font-bold mb-2">📸 AFTER IMAGE</h3>
                      {afterImage ? (
                        <div className="matrix-border rounded overflow-hidden">
                          <img
                            src={URL.createObjectURL(afterImage)}
                            alt="After"
                            className="w-full h-48 object-cover"
                          />
                        </div>
                      ) : (
                        <div className="matrix-border rounded bg-black/50 h-48 flex items-center justify-center">
                          <span className="matrix-text opacity-50">No image selected</span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* Execute Button */}
              <Button
                onClick={handleAnalysis}
                disabled={!beforeImage || !afterImage || isProcessing}
                className="w-full matrix-button text-xl py-6 font-mono font-bold tracking-wider"
              >
                {isProcessing ? (
                  <span className="matrix-pulse">
                    {analysisMode === "agent_orchestrated" 
                      ? "&gt; AI AGENT ANALYZING... &lt;" 
                      : "&gt; PROCESSING SATELLITE DATA... &lt;"
                    }
                  </span>
                ) : (
                  analysisMode === "agent_orchestrated" 
                    ? "&gt; EXECUTE AI AGENT ANALYSIS &lt;"
                    : "&gt; EXECUTE CHANGE DETECTION &lt;"
                )}
              </Button>
            </CardContent>
          </Card>

          {/* Right Panel: Results */}
          <Card className="matrix-card matrix-glow">
            <CardHeader className="matrix-scanline">
              <CardTitle className="matrix-text text-2xl font-mono">
                {'>>'} AI AGENT ANALYSIS RESULTS
              </CardTitle>
              <CardDescription className="matrix-text opacity-70">
                AI agent orchestration output and comprehensive analysis matrix
              </CardDescription>
            </CardHeader>
            <CardContent>
              {results ? (
                <div className="space-y-6">
                  {results.success ? (
                    <>
                      {/* Results Method Banner */}
                      <div className="matrix-border p-3 rounded-lg bg-green-900/20">
                        <div className="matrix-text text-center font-mono">
                          {('orchestration_method' in results && results.orchestration_method?.includes('agent')) 
                            ? "🤖 AI AGENT ORCHESTRATED ANALYSIS COMPLETE"
                            : "🔍 DIRECT SATELLITE ANALYSIS COMPLETE"
                          }
                        </div>
                        <div className="matrix-text text-center font-mono text-sm opacity-70 mt-1">
                          {('tools_used' in results && results.tools_used) 
                            ? `TOOLS USED: ${results.tools_used.map(tool => tool.toUpperCase()).join(' • ')}`
                            : "OPENCV PIXEL DETECTION • GPT-4 VISION ANALYSIS"
                          }
                        </div>
                      </div>

                      {/* Agent Analysis (Primary for agent mode) */}
                      {('agent_analysis' in results && results.agent_analysis) && (
                        <div className="matrix-border p-4 rounded-lg bg-purple-900/20">
                          <h3 className="matrix-text text-xl font-mono mb-4">
                            [🤖 AI AGENT COMPREHENSIVE ANALYSIS]
                          </h3>
                          <div className="matrix-border p-4 rounded bg-black/50">
                            <p className="matrix-text text-sm leading-relaxed whitespace-pre-wrap">
                              {results.agent_analysis}
                            </p>
                          </div>
                          <div className="mt-2 text-xs matrix-text opacity-60">
                            Powered by OpenAI Agent with MCP Tools
                          </div>
                        </div>
                      )}

                      {/* Significance Assessment (from agent) */}
                      {(() => {
                        const assessment = getSignificanceAssessment(results as AgentResults);
                        if (!assessment) return null;
                        
                        return (
                          <div className={`matrix-border p-4 rounded-lg ${
                            assessment.significance_level === 'HIGH' ? 'bg-red-900/30 border-red-400' :
                            assessment.significance_level === 'MEDIUM' ? 'bg-yellow-900/30 border-yellow-400' :
                            'bg-green-900/10'
                          }`}>
                            <div className="text-center">
                              <div className={`text-2xl font-bold font-mono ${
                                assessment.significance_level === 'HIGH' ? 'text-red-400' :
                                assessment.significance_level === 'MEDIUM' ? 'text-yellow-400' :
                                'text-green-400'
                              }`}>
                                SIGNIFICANCE: {assessment.significance_level}
                              </div>
                              <div className="matrix-text text-lg mt-2">
                                URGENCY: {assessment.urgency?.replace(/_/g, ' ')}
                              </div>
                              <div className="matrix-text text-sm mt-2 opacity-80">
                                PATTERN: {assessment.change_pattern?.replace(/_/g, ' ')}
                              </div>
                            </div>
                            {assessment.recommendation && (
                              <div className="matrix-border p-3 rounded bg-black/50 mt-4">
                                <div className="matrix-text text-sm">
                                  <span className="text-green-400">RECOMMENDATION:</span> {assessment.recommendation}
                                </div>
                              </div>
                            )}
                          </div>
                        );
                      })()}

                      {/* Change Detection Metrics */}
                      {(() => {
                        const changeData = getChangeDetectionData(results);
                        if (!changeData) return null;

                        return (
                          <div className="matrix-border p-4 rounded-lg bg-green-900/10">
                            <h3 className="matrix-text text-xl font-mono mb-4">
                              [📊 CHANGE DETECTION METRICS]
                            </h3>
                            
                            {/* Change Percentage - Most Important Metric */}
                            <div className="matrix-border p-4 rounded-lg bg-black/50 mb-4">
                              <div className="text-center">
                                <div className="matrix-text text-5xl font-bold matrix-pulse mb-2">
                                  {(changeData.change_percentage || 0).toFixed(2)}%
                                </div>
                                <div className="matrix-text text-lg opacity-80">OF IMAGE AREA CHANGED</div>
                                <div className="matrix-text text-sm opacity-60 mt-1">
                                  {changeData.change_percentage && changeData.change_percentage > 5 
                                    ? "🔴 SIGNIFICANT CHANGE DETECTED" 
                                    : changeData.change_percentage > 1 
                                    ? "🟡 MODERATE CHANGE DETECTED"
                                    : "🟢 MINIMAL CHANGE DETECTED"
                                  }
                                </div>
                              </div>
                            </div>

                            {/* Detailed Metrics Grid */}
                            <div className="grid grid-cols-2 gap-4 mb-4">
                              <div className="matrix-border p-3 rounded bg-black/50">
                                <div className="matrix-text text-2xl font-bold matrix-pulse text-center">
                                  {changeData.contours_count || 0}
                                </div>
                                <div className="matrix-text text-sm opacity-70 text-center">CHANGE REGIONS</div>
                              </div>
                              <div className="matrix-border p-3 rounded bg-black/50">
                                <div className="matrix-text text-lg font-bold text-center">
                                  {changeData.analysis_method?.toUpperCase() || 'OPENCV'}
                                </div>
                                <div className="matrix-text text-sm opacity-70 text-center">ANALYSIS METHOD</div>
                              </div>
                            </div>

                            {/* Pixel Details */}
                            <div className="matrix-border p-3 rounded bg-black/50">
                              <div className="matrix-text font-mono text-sm">
                                PIXELS CHANGED: {(changeData.changed_pixels || 0).toLocaleString()} / {(changeData.total_pixels || 0).toLocaleString()}
                              </div>
                            </div>
                          </div>
                        );
                      })()}

                      {/* GPT-4 Vision Analysis */}
                      {(() => {
                        const visionAnalysis = getVisionAnalysis(results);
                        if (!visionAnalysis) return null;

                        return (
                          <div className="matrix-border p-4 rounded-lg bg-blue-900/20">
                            <h3 className="matrix-text text-xl font-mono mb-4">
                              [👁️ GPT-4 VISION ANALYSIS]
                            </h3>
                            <div className="matrix-border p-4 rounded bg-black/50">
                              <p className="matrix-text text-sm leading-relaxed whitespace-pre-wrap">
                                {visionAnalysis}
                              </p>
                            </div>
                            <div className="mt-2 text-xs matrix-text opacity-60">
                              {('model_used' in results && results.model_used) 
                                ? `Powered by ${results.model_used}`
                                : 'Powered by GPT-4 Vision'
                              }
                            </div>
                          </div>
                        );
                      })()}

                      {/* Change Mask Visualization */}
                      {(() => {
                        const changeData = getChangeDetectionData(results);
                        if (!changeData?.change_mask_base64) return null;

                        return (
                          <div className="matrix-border p-4 rounded-lg bg-green-900/10">
                            <h3 className="matrix-text text-xl font-mono mb-4">
                              [🎯 VISUAL CHANGE MATRIX]
                            </h3>
                            <div className="matrix-border rounded-lg overflow-hidden matrix-glow">
                              <img
                                src={`data:image/png;base64,${changeData.change_mask_base64}`}
                                alt="Change Detection Mask"
                                className="w-full h-auto"
                              />
                            </div>
                            <div className="mt-2 text-xs matrix-text opacity-60 text-center">
                              Green regions indicate detected changes
                            </div>
                          </div>
                        );
                      })()}

                      {/* Tool Results Debug (for agent mode) */}
                      {('tool_results' in results && results.tool_results && results.tool_results.length > 0) && (
                        <div className="matrix-border p-4 rounded-lg bg-gray-900/10">
                          <h3 className="matrix-text text-xl font-mono mb-4">
                            [🛠️ TOOL EXECUTION DETAILS]
                          </h3>
                          <div className="space-y-2">
                            {results.tool_results.map((tool, index) => (
                              <div key={index} className="matrix-border p-3 rounded bg-black/50">
                                <div className="flex justify-between items-center">
                                  <div className="matrix-text font-mono">
                                    {tool.tool_name.replace(/_/g, ' ').toUpperCase()}
                                  </div>
                                  <div className={`font-mono text-sm ${
                                    tool.result.success ? 'text-green-400' : 'text-red-400'
                                  }`}>
                                    {tool.result.success ? '✅ SUCCESS' : '❌ FAILED'}
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </>
                  ) : (
                    /* Error Display */
                    <div className="matrix-border p-4 rounded-lg bg-red-900/20 border-red-400">
                      <h3 className="matrix-text text-xl font-mono mb-4 text-red-400">
                        [❌ ANALYSIS FAILED]
                      </h3>
                      <div className="matrix-border p-4 rounded bg-black/50">
                        <p className="matrix-text text-sm">
                          {results.error || 'Unknown error occurred during analysis'}
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                /* No Results Yet */
                <div className="matrix-border p-8 rounded-lg text-center bg-blue-900/10">
                  <div className="matrix-text opacity-50 font-mono">
                    {analysisMode === "agent_orchestrated" 
                      ? "🤖 AI AGENT READY FOR SATELLITE ANALYSIS"
                      : "🔍 MATRIX SYSTEM READY FOR CHANGE DETECTION"
                    }
                  </div>
                  <div className="matrix-text opacity-30 font-mono text-sm mt-2">
                    Upload images and execute analysis to see results
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Footer */}
        <div className="text-center mt-12">
          <div className="matrix-text opacity-50 font-mono text-sm">
            POWERED BY: OpenCV Neural Networks • GPT-4 Vision AI • MCP Protocol v2.0 • FastAPI Matrix Interface
          </div>
          <div className="matrix-text opacity-30 font-mono text-xs mt-2">
            "There is no spoon. Only data." - The Matrix Change Detection Protocol
          </div>
        </div>
      </div>
    </div>
  );
}
