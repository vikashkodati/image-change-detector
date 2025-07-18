"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
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

// Sample high-resolution disaster imagery optimized for CLIP semantic analysis
const sampleImages: SampleImage[] = [
  {
    id: 1,
    name: "Hurricane Ian - Power Grid Analysis",
    before: "/samples/hurricane_ian_before.png",
    after: "/samples/hurricane_ian_after.png",
    description: "Hurricane Ian impact on Florida's power grid - nighttime lights before/after (NASA, 2022)",
    resolution: "7680x2160 (NASA Black Marble)",
    clipEngagement: "HIGH - Semantic analysis of infrastructure damage patterns"
  },
  {
    id: 2,
    name: "Amazon Deforestation Monitoring",
    before: "/samples/hurricane_ian_before.png", // Temporary placeholder
    after: "/samples/hurricane_ian_after.png",   // Temporary placeholder
    description: "Amazon rainforest clearing patterns - CLIP detects forest-to-farmland conversion (Landsat-8, 2024)",
    resolution: "Landsat-8 30m resolution",
    clipEngagement: "VERY HIGH - Semantic understanding of land use change from forest to agriculture"
  },
  {
    id: 3,
    name: "Urban Development Expansion",
    before: "/samples/la_wildfire_current.jpg",  // Temporary placeholder
    after: "/samples/hurricane_ian_after.png",   // Temporary placeholder
    description: "City sprawl detection - suburban expansion into rural areas (WorldView-3, 2024)",
    resolution: "WorldView-3 0.5m resolution", 
    clipEngagement: "HIGH - Semantic classification of residential vs. agricultural land patterns"
  },
  {
    id: 4,
    name: "Flood Impact Assessment",
    before: "/samples/hurricane_ian_before.png", // Temporary placeholder
    after: "/samples/la_wildfire_current.jpg",   // Temporary placeholder
    description: "Major flooding event - water level changes and infrastructure impact (Sentinel-1 SAR, 2024)",
    resolution: "Sentinel-1 SAR 10m resolution",
    clipEngagement: "VERY HIGH - Semantic detection of water vs. land, flooded infrastructure"
  },
  {
    id: 5,
    name: "Agricultural Field Rotation",
    before: "/samples/la_wildfire_current.jpg",  // Temporary placeholder
    after: "/samples/hurricane_ian_before.png",  // Temporary placeholder
    description: "Crop rotation patterns - semantic analysis of different agricultural land use (PlanetScope, 2024)",
    resolution: "PlanetScope 3m resolution",
    clipEngagement: "MEDIUM - CLIP identifies different crop types and growth stages"
  },
  {
    id: 6,
    name: "Wildfire Smoke Detection",
    before: "/samples/la_wildfire_current.jpg",
    after: "/samples/la_wildfire_current.jpg",
    description: "Los Angeles wildfire smoke captured by Sentinel-2 (ESA, January 2025)",
    resolution: "Sentinel-2 10m resolution",
    clipEngagement: "MEDIUM - Visual smoke pattern analysis vs. clear skies"
  }
];

export default function Home() {
  const [beforeImage, setBeforeImage] = useState<File | null>(null);
  const [afterImage, setAfterImage] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState<{
    success: boolean;
    method?: string;
    results?: {
      // OpenCV results (legacy compatibility)
      change_percentage?: number;
      changed_pixels?: number;
      total_pixels?: number;
      change_mask_base64?: string;
      contours_count?: number;
      
      // Hybrid results structure
      hybrid_available?: boolean;
      opencv_results?: {
        change_percentage: number;
        changed_pixels: number;
        total_pixels: number;
        change_mask_base64: string;
        contours_count: number;
      };
      semantic_results?: {
        available: boolean;
        semantic_similarity?: number;
        is_meaningful_change?: boolean;
        change_confidence?: number;
        interpretation?: string;
        error?: string;
      };
      classification_results?: {
        available: boolean;
        top_categories?: Array<{
          category: string;
          confidence: number;
          likelihood: string;
        }>;
        most_likely?: string;
        max_confidence?: number;
        error?: string;
      };
      final_assessment?: {
        has_meaningful_change: boolean;
        confidence: number;
        reasoning: string;
        change_type: string;
        threat_level: string;
      };
      processing_time?: {
        opencv_stage: string;
        clip_stage: string;
      };
    };
    enhanced_features?: {
      semantic_analysis: boolean;
      change_classification: boolean;
      threat_assessment: string;
    };
    error?: string;
  } | null>(null);
  const [selectedSample, setSelectedSample] = useState<number | null>(null);

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

  const handleDetectChanges = async () => {
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
        }),
      });

      const data = await response.json();
      setResults(data);
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
              🕶️ MCP-POWERED SATELLITE IMAGE ANALYSIS SYSTEM 🕶️
            </p>
          </div>
        </div>

        {/* Enhanced MCP Status Banner */}
        <div className="matrix-card matrix-glow p-4 mb-8 rounded-lg matrix-scanline">
          <div className="flex items-center justify-center space-x-2 flex-wrap">
            <div className="matrix-pulse">
              <div className="w-3 h-3 bg-green-400 rounded-full"></div>
            </div>
            <span className="matrix-text font-mono text-lg">
              [MATRIX AI: ONLINE] • MCP TOOLS: ACTIVE • HYBRID DETECTION: ENABLED
            </span>
            <div className="matrix-pulse">
              <div className="w-3 h-3 bg-green-400 rounded-full"></div>
            </div>
          </div>
          <div className="flex items-center justify-center space-x-2 mt-2 flex-wrap">
            <span className="matrix-text font-mono text-sm opacity-80">
              🧠 OpenCV + CLIP Semantic Analysis • 🎯 Change Classification • ⚡ Threat Assessment
            </span>
          </div>
          {/* CLIP Debug Panel */}
          <div className="mt-4 flex justify-center space-x-4">
            <Button
              variant="outline"
              size="sm"
              onClick={async () => {
                try {
                  const response = await fetch(`${API_URL}/api/clip-status`);
                  const data = await response.json();
                  alert(`CLIP Status:\n${JSON.stringify(data, null, 2)}`);
                } catch (error) {
                  alert(`Error checking CLIP status: ${error}`);
                }
              }}
              className="text-xs"
            >
              🔍 CHECK CLIP STATUS
            </Button>
            <Button
              variant="outline" 
              size="sm"
              onClick={async () => {
                try {
                  const response = await fetch(`${API_URL}/api/force-clip-init`, { method: 'POST' });
                  const data = await response.json();
                  alert(`Force CLIP Init:\n${JSON.stringify(data, null, 2)}`);
                } catch (error) {
                  alert(`Error forcing CLIP init: ${error}`);
                }
              }}
              className="text-xs"
            >
              🚀 FORCE CLIP INIT
            </Button>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          
          {/* Left Panel: Upload Interface */}
          <Card className="matrix-card matrix-glow">
            <CardHeader className="matrix-scanline">
              <CardTitle className="matrix-text text-2xl font-mono">
                {'>>'} UPLOAD SURVEILLANCE DATA
              </CardTitle>
              <CardDescription className="matrix-text opacity-70">
                Insert satellite imagery for deep analysis via MCP neural network
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Sample Data Section */}
              <div>
                <Label className="matrix-text text-lg font-mono block mb-4">
                  &gt; CLASSIFIED SAMPLE DATA - CLIP AI TESTING:
                </Label>
                <div className="matrix-border p-3 rounded-lg bg-blue-900/10 mb-4">
                  <p className="matrix-text font-mono text-sm opacity-80">
                    🧠 <span className="text-blue-400">AI ENGAGEMENT LEVELS:</span> Each sample is optimized to test different CLIP semantic analysis capabilities
                  </p>
                  <div className="flex flex-wrap gap-2 mt-2">
                    <span className="px-2 py-1 bg-red-900/50 text-red-300 border border-red-500 rounded text-xs font-mono">
                      VERY HIGH: Complex semantic understanding required
                    </span>
                    <span className="px-2 py-1 bg-orange-900/50 text-orange-300 border border-orange-500 rounded text-xs font-mono">
                      HIGH: Advanced pattern recognition
                    </span>
                    <span className="px-2 py-1 bg-yellow-900/50 text-yellow-300 border border-yellow-500 rounded text-xs font-mono">
                      MEDIUM: Standard semantic analysis
                    </span>
                  </div>
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
                      {/* CLIP Engagement Badge */}
                      <div className="flex justify-between items-start mb-2">
                        <h3 className="matrix-text font-mono font-bold text-lg">
                          {sample.name}
                        </h3>
                        <div className={`px-2 py-1 rounded text-xs font-mono font-bold ${
                          sample.clipEngagement?.startsWith('VERY HIGH') ? 'bg-red-900/50 text-red-300 border border-red-500' :
                          sample.clipEngagement?.startsWith('HIGH') ? 'bg-orange-900/50 text-orange-300 border border-orange-500' :
                          'bg-yellow-900/50 text-yellow-300 border border-yellow-500'
                        }`}>
                          🧠 {sample.clipEngagement?.split(' - ')[0] || 'UNKNOWN'}
                        </div>
                      </div>
                      
                      <p className="matrix-text opacity-70 text-sm mb-2">
                        {sample.description}
                      </p>
                      
                      {/* CLIP Analysis Description */}
                      <div className="matrix-border rounded p-2 mb-2 bg-green-900/10">
                        <p className="matrix-text opacity-60 text-xs font-mono">
                          🔍 CLIP ANALYSIS: {sample.clipEngagement?.split(' - ')[1] || 'Standard semantic analysis'}
                        </p>
                      </div>
                      
                      <p className="matrix-text opacity-50 text-xs font-mono">
                        RES: {sample.resolution}
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

              {/* Execute Button */}
              <Button
                onClick={handleDetectChanges}
                disabled={!beforeImage || !afterImage || isProcessing}
                className="w-full matrix-button text-xl py-6 font-mono font-bold tracking-wider"
              >
                {isProcessing ? (
                  <span className="matrix-pulse">
                    &gt; ANALYZING... NEURAL NETWORK PROCESSING &lt;
                  </span>
                ) : (
                  "&gt; EXECUTE MCP ANALYSIS &lt;"
                )}
              </Button>
            </CardContent>
          </Card>

          {/* Right Panel: Results */}
          <Card className="matrix-card matrix-glow">
            <CardHeader className="matrix-scanline">
              <CardTitle className="matrix-text text-2xl font-mono">
                {'>>'} ANALYSIS RESULTS
              </CardTitle>
              <CardDescription className="matrix-text opacity-70">
                MCP neural network output and change detection matrix
              </CardDescription>
            </CardHeader>
            <CardContent>
              {results ? (
                <div className="space-y-6">
                  {results.success && results.results ? (
                    <>
                      {/* Method Detection Banner */}
                      <div className={`matrix-border p-3 rounded-lg ${
                        results.method === 'hybrid_detection' ? 'bg-green-900/20' : 'bg-blue-900/20'
                      }`}>
                        <div className="matrix-text text-center font-mono">
                          {results.method === 'hybrid_detection' ? 
                            '🧠 HYBRID AI ANALYSIS: OpenCV + CLIP Semantic Detection' : 
                            '🔍 LEGACY ANALYSIS: OpenCV Only'
                          }
                        </div>
                      </div>

                      {/* Enhanced Hybrid Results */}
                      {results.results.hybrid_available && results.results.final_assessment ? (
                        <>
                          {/* Threat Assessment Banner */}
                          <div className={`matrix-border p-4 rounded-lg ${
                            results.results.final_assessment.threat_level === 'HIGH' ? 'bg-red-900/30 border-red-400' :
                            results.results.final_assessment.threat_level === 'MEDIUM' ? 'bg-yellow-900/30 border-yellow-400' :
                            'bg-green-900/10'
                          }`}>
                            <div className="text-center">
                              <div className={`text-2xl font-bold font-mono ${
                                results.results.final_assessment.threat_level === 'HIGH' ? 'text-red-400' :
                                results.results.final_assessment.threat_level === 'MEDIUM' ? 'text-yellow-400' :
                                'text-green-400'
                              }`}>
                                THREAT LEVEL: {results.results.final_assessment.threat_level}
                              </div>
                              <div className="matrix-text text-lg mt-2">
                                CONFIDENCE: {((results.results.final_assessment.confidence || 0) * 100).toFixed(0)}%
                              </div>
                            </div>
                          </div>

                          {/* Final Assessment */}
                          <div className="matrix-border p-4 rounded-lg bg-green-900/10">
                            <h3 className="matrix-text text-xl font-mono mb-4">
                              [AI ASSESSMENT MATRIX]
                            </h3>
                            <div className="space-y-3">
                              <div className="matrix-border p-3 rounded bg-black/50">
                                <div className="matrix-text font-mono">
                                  <span className="text-green-400">STATUS:</span> {
                                    results.results.final_assessment.has_meaningful_change ? 
                                    'MEANINGFUL CHANGE DETECTED' : 'NO SIGNIFICANT CHANGE'
                                  }
                                </div>
                              </div>
                              <div className="matrix-border p-3 rounded bg-black/50">
                                <div className="matrix-text font-mono">
                                  <span className="text-green-400">TYPE:</span> {results.results.final_assessment.change_type?.toUpperCase() || 'UNKNOWN'}
                                </div>
                              </div>
                              <div className="matrix-border p-3 rounded bg-black/50">
                                <div className="matrix-text font-mono text-sm">
                                  <span className="text-green-400">ANALYSIS:</span> {results.results.final_assessment.reasoning || 'No analysis available'}
                                </div>
                              </div>
                            </div>
                          </div>

                          {/* OpenCV Pixel Analysis */}
                          {results.results.opencv_results && (
                            <div className="matrix-border p-4 rounded-lg bg-blue-900/10">
                              <h3 className="matrix-text text-xl font-mono mb-4">
                                [PIXEL DETECTION MATRIX]
                              </h3>
                              <div className="grid grid-cols-2 gap-4 text-center">
                                <div className="matrix-border p-3 rounded bg-black/50">
                                  <div className="matrix-text text-3xl font-bold matrix-pulse">
                                    {(results.results.opencv_results.change_percentage || 0).toFixed(2)}%
                                  </div>
                                  <div className="matrix-text text-sm opacity-70">PIXEL CHANGE</div>
                                </div>
                                <div className="matrix-border p-3 rounded bg-black/50">
                                  <div className="matrix-text text-3xl font-bold matrix-pulse">
                                    {results.results.opencv_results.contours_count || 0}
                                  </div>
                                  <div className="matrix-text text-sm opacity-70">REGIONS</div>
                                </div>
                              </div>
                              <div className="mt-4 matrix-border p-3 rounded bg-black/50">
                                <div className="matrix-text font-mono text-sm">
                                  PIXELS: {(results.results.opencv_results.changed_pixels || 0).toLocaleString()} / {(results.results.opencv_results.total_pixels || 0).toLocaleString()}
                                </div>
                              </div>
                            </div>
                          )}

                          {/* Semantic Analysis */}
                          {results.results.semantic_results?.available && (
                            <div className="matrix-border p-4 rounded-lg bg-purple-900/10">
                              <h3 className="matrix-text text-xl font-mono mb-4">
                                [SEMANTIC ANALYSIS MATRIX]
                              </h3>
                              <div className="space-y-3">
                                <div className="matrix-border p-3 rounded bg-black/50">
                                  <div className="matrix-text font-mono">
                                    <span className="text-green-400">SIMILARITY:</span> {
                                      ((results.results.semantic_results.semantic_similarity || 0) * 100).toFixed(1)
                                    }%
                                  </div>
                                </div>
                                {results.results.semantic_results.interpretation && (
                                  <div className="matrix-border p-3 rounded bg-black/50">
                                    <div className="matrix-text font-mono text-sm">
                                      <span className="text-green-400">INTERPRETATION:</span> {results.results.semantic_results.interpretation}
                                    </div>
                                  </div>
                                )}
                              </div>
                            </div>
                          )}

                          {/* Change Classification */}
                          {results.results.classification_results?.available && results.results.classification_results.top_categories && (
                            <div className="matrix-border p-4 rounded-lg bg-orange-900/10">
                              <h3 className="matrix-text text-xl font-mono mb-4">
                                [CHANGE CLASSIFICATION MATRIX]
                              </h3>
                              <div className="space-y-2">
                                {results.results.classification_results.top_categories.slice(0, 3).map((category, index) => (
                                  <div key={index} className="matrix-border p-3 rounded bg-black/50">
                                    <div className="flex justify-between items-center">
                                      <div className="matrix-text font-mono">
                                        {category.category?.toUpperCase() || 'UNKNOWN'}
                                      </div>
                                      <div className={`font-mono text-sm ${
                                        category.likelihood === 'high' ? 'text-green-400' :
                                        category.likelihood === 'medium' ? 'text-yellow-400' : 'text-red-400'
                                      }`}>
                                        {((category.confidence || 0) * 100).toFixed(1)}% ({category.likelihood?.toUpperCase() || 'UNKNOWN'})
                                      </div>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}

                          {/* Change Mask Visualization */}
                          {results.results.opencv_results?.change_mask_base64 && (
                            <div className="matrix-border p-4 rounded-lg bg-green-900/10">
                              <h3 className="matrix-text text-xl font-mono mb-4">
                                [VISUAL CHANGE MATRIX]
                              </h3>
                              <div className="matrix-border rounded-lg overflow-hidden matrix-glow">
                                <img
                                  src={`data:image/png;base64,${results.results.opencv_results.change_mask_base64}`}
                                  alt="Change Detection Mask"
                                  className="w-full h-auto"
                                />
                              </div>
                            </div>
                          )}

                          {/* Processing Performance */}
                          {results.results.processing_time && (
                            <div className="matrix-border p-4 rounded-lg bg-gray-900/10">
                              <h3 className="matrix-text text-xl font-mono mb-4">
                                [PROCESSING PERFORMANCE]
                              </h3>
                              <div className="grid grid-cols-2 gap-4">
                                <div className="matrix-border p-3 rounded bg-black/50">
                                  <div className="matrix-text font-mono text-sm">
                                    <span className="text-green-400">OPENCV:</span> {results.results.processing_time?.opencv_stage || 'N/A'}
                                  </div>
                                </div>
                                <div className="matrix-border p-3 rounded bg-black/50">
                                  <div className="matrix-text font-mono text-sm">
                                    <span className="text-green-400">CLIP:</span> {results.results.processing_time?.clip_stage || 'N/A'}
                                  </div>
                                </div>
                              </div>
                            </div>
                          )}
                        </>
                      ) : (
                        /* Legacy Results Display (backward compatibility) */
                        <>
                          <div className="matrix-border p-4 rounded-lg bg-green-900/10">
                            <h3 className="matrix-text text-xl font-mono mb-4">
                              [CHANGE DETECTION MATRIX]
                            </h3>
                            <div className="grid grid-cols-2 gap-4 text-center">
                              <div className="matrix-border p-3 rounded bg-black/50">
                                <div className="matrix-text text-3xl font-bold matrix-pulse">
                                  {(results.results.change_percentage || 0).toFixed(2)}%
                                </div>
                                <div className="matrix-text text-sm opacity-70">CHANGE RATE</div>
                              </div>
                              <div className="matrix-border p-3 rounded bg-black/50">
                                <div className="matrix-text text-3xl font-bold matrix-pulse">
                                  {results.results.contours_count || 0}
                                </div>
                                <div className="matrix-text text-sm opacity-70">ANOMALIES</div>
                              </div>
                            </div>
                            
                            <div className="mt-4 matrix-border p-3 rounded bg-black/50">
                              <div className="matrix-text text-lg font-mono">
                                PIXELS ALTERED: {(results.results.changed_pixels || 0).toLocaleString()} / {(results.results.total_pixels || 0).toLocaleString()}
                              </div>
                            </div>
                          </div>

                          {results.results.change_mask_base64 && (
                            <div className="matrix-border p-4 rounded-lg bg-green-900/10">
                              <h3 className="matrix-text text-xl font-mono mb-4">
                                [CHANGE MASK OVERLAY]
                              </h3>
                              <div className="matrix-border rounded-lg overflow-hidden matrix-glow">
                                <img
                                  src={`data:image/png;base64,${results.results.change_mask_base64}`}
                                  alt="Change Detection Mask"
                                  className="w-full h-auto"
                                />
                              </div>
                            </div>
                          )}
                        </>
                      )}
                    </>
                  ) : (
                    <div className="matrix-border p-6 rounded-lg bg-red-900/20 border-red-400">
                      <h3 className="matrix-text text-xl font-mono mb-2 text-red-400">
                        [SYSTEM ERROR]
                      </h3>
                      <p className="matrix-text text-red-300">
                        {results.error || "Unknown error occurred during neural network processing"}
                      </p>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-12">
                  <div className="matrix-text opacity-50 text-xl font-mono">
                    &gt; AWAITING INPUT DATA
                  </div>
                  <div className="matrix-text opacity-30 text-sm mt-4 font-mono">
                    Upload surveillance imagery to begin MCP analysis...
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
