"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

// Sample high-resolution disaster imagery
const sampleImages = [
  {
    id: 1,
    name: "Hurricane Ian - Florida Power Grid",
    before: "/samples/hurricane_ian_before.png",
    after: "/samples/hurricane_ian_after.png",
    description: "Hurricane Ian impact on Florida's power grid - nighttime lights before/after (NASA, 2022)",
    resolution: "7680x2160 (NASA Black Marble)"
  },
  {
    id: 2,
    name: "Los Angeles Wildfires",
    before: "/samples/la_wildfire_current.jpg",
    after: "/samples/la_wildfire_current.jpg", // Same image for now
    description: "Los Angeles wildfire smoke captured by Sentinel-2 (ESA, January 2025)",
    resolution: "Sentinel-2 10m resolution"
  }
];

export default function Home() {
  const [beforeImage, setBeforeImage] = useState<File | null>(null);
  const [afterImage, setAfterImage] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [selectedSample, setSelectedSample] = useState<number | null>(null);

  // Convert File to base64 string
  const fileToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => {
        const result = reader.result as string;
        // Remove data URL prefix (data:image/jpeg;base64,)
        const base64 = result.split(',')[1];
        resolve(base64);
      };
      reader.onerror = error => reject(error);
    });
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>, type: 'before' | 'after') => {
    const file = event.target.files?.[0];
    if (file) {
      if (type === 'before') {
        setBeforeImage(file);
      } else {
        setAfterImage(file);
      }
      // Clear selected sample when user uploads their own files
      setSelectedSample(null);
    }
  };

  // Convert URL to File object
  const urlToFile = async (url: string, filename: string): Promise<File> => {
    const response = await fetch(url);
    const blob = await response.blob();
    return new File([blob], filename, { type: blob.type });
  };

  const handleSampleSelect = async (sample: typeof sampleImages[0]) => {
    try {
      setIsProcessing(true);
      
      // Convert sample images to File objects
      const beforeFile = await urlToFile(sample.before, `${sample.name}_before`);
      const afterFile = await urlToFile(sample.after, `${sample.name}_after`);
      
      setBeforeImage(beforeFile);
      setAfterImage(afterFile);
      setSelectedSample(sample.id);
      setResults(null); // Clear previous results
    } catch (error) {
      console.error('Error loading sample images:', error);
      alert('Failed to load sample images');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleAnalyze = async () => {
    if (!beforeImage || !afterImage) {
      alert('Please upload both before and after images');
      return;
    }
    
    setIsProcessing(true);
    
    try {
      // Convert images to base64
      const beforeBase64 = await fileToBase64(beforeImage);
      const afterBase64 = await fileToBase64(afterImage);
      
      // Call REST API server to detect changes
      const response = await fetch('http://127.0.0.1:8000/api/detect-changes', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          before_image_base64: beforeBase64,
          after_image_base64: afterBase64
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Handle REST API response format
      if (!data.success) {
        throw new Error(data.error || 'API call failed');
      }
      setResults(data);
      
    } catch (error) {
      console.error('Error analyzing images:', error);
      alert('Failed to analyze images. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-violet-900 p-8">
      <div className="max-w-6xl mx-auto">
        {/* MCP Status Banner */}
        <div className="mb-6 bg-gradient-to-r from-green-500/20 to-emerald-500/20 border border-green-500/30 rounded-lg p-4">
          <div className="flex items-center justify-center space-x-2">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
            <span className="text-green-300 font-medium">
              🚀 MCP-Powered Backend Active - FastMCP Tools Available
            </span>
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
          </div>
        </div>

        <div className="text-center mb-12">
          <h1 className="text-6xl font-bold bg-gradient-to-r from-white via-purple-100 to-violet-100 bg-clip-text text-transparent mb-4">
            Image Change Detector
          </h1>
          <p className="text-xl text-gray-300 max-w-2xl mx-auto leading-relaxed">
            AI-powered satellite image analysis using OpenCV computer vision and GPT-4 Vision. 
            Detect changes, analyze impacts, and get instant insights.
          </p>
        </div>

        <Card className="mb-8 bg-gray-800/50 border-gray-700/50 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="text-white text-2xl font-bold">Upload Images</CardTitle>
            <CardDescription className="text-gray-300 text-lg">
              Select before and after satellite images to compare with AI-powered analysis
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-2">
                <Label htmlFor="before-image" className="text-gray-200 font-medium">Before Image</Label>
                <Input
                  id="before-image"
                  type="file"
                  accept="image/*,.tiff,.tif,.geotiff"
                  onChange={(e) => handleFileUpload(e, 'before')}
                  className="bg-gray-700/50 border-gray-600 text-gray-200 file:bg-purple-600 file:text-white file:border-0 file:rounded-md"
                />
                {beforeImage && (
                  <p className="text-sm text-green-400 flex items-center">
                    <span className="w-2 h-2 bg-green-400 rounded-full mr-2"></span>
                    {beforeImage.name} uploaded
                  </p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="after-image" className="text-gray-200 font-medium">After Image</Label>
                <Input
                  id="after-image"
                  type="file"
                  accept="image/*,.tiff,.tif,.geotiff"
                  onChange={(e) => handleFileUpload(e, 'after')}
                  className="bg-gray-700/50 border-gray-600 text-gray-200 file:bg-purple-600 file:text-white file:border-0 file:rounded-md"
                />
                {afterImage && (
                  <p className="text-sm text-green-400 flex items-center">
                    <span className="w-2 h-2 bg-green-400 rounded-full mr-2"></span>
                    {afterImage.name} uploaded
                  </p>
                )}
              </div>
            </div>

            <div className="mt-8 text-center">
              <Button 
                onClick={handleAnalyze}
                disabled={!beforeImage || !afterImage || isProcessing}
                className="px-12 py-4 text-lg font-bold bg-gradient-to-r from-purple-600 to-violet-600 hover:from-purple-700 hover:to-violet-700 border-0 rounded-xl shadow-lg transform transition-all duration-200 hover:scale-105"
              >
                {isProcessing ? (
                  <span className="flex items-center">
                    <div className="animate-spin w-5 h-5 border-2 border-white border-t-transparent rounded-full mr-3"></div>
                    Analyzing with AI...
                  </span>
                ) : (
                  <span className="flex items-center">
                    🔍 Analyze Changes with MCP
                  </span>
                )}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Sample Images Section */}
        <Card className="mb-8 bg-gray-800/50 border-gray-700/50 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="text-white text-2xl font-bold">Sample Disaster Imagery</CardTitle>
            <CardDescription className="text-gray-300 text-lg">
              Try the app with high-resolution satellite images from real disasters
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {sampleImages.map((sample) => (
                <div
                  key={sample.id}
                  className={`border rounded-xl p-6 cursor-pointer transition-all duration-300 ${
                    selectedSample === sample.id
                      ? 'border-purple-500 bg-gradient-to-br from-purple-500/20 to-violet-500/20 shadow-lg transform scale-105'
                      : 'border-gray-600 bg-gray-700/30 hover:border-purple-400 hover:bg-gray-700/50 hover:shadow-md'
                  }`}
                  onClick={() => handleSampleSelect(sample)}
                >
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="font-bold text-white text-base">{sample.name}</h3>
                    {selectedSample === sample.id && (
                      <span className="text-xs bg-gradient-to-r from-purple-500 to-violet-500 text-white px-3 py-1 rounded-full font-medium">
                        ✓ Selected
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-gray-300 mb-3 leading-relaxed">{sample.description}</p>
                  <p className="text-xs text-gray-400 font-medium">{sample.resolution}</p>
                  
                  {/* Image preview */}
                  <div className="mt-3 flex space-x-2">
                    <div className="flex-1">
                      <img
                        src={sample.before}
                        alt={`${sample.name} before`}
                        className="w-full h-24 object-cover rounded-lg border border-gray-600"
                      />
                      <p className="text-xs text-center mt-2 text-gray-300 font-medium">Before</p>
                    </div>
                    <div className="flex-1">
                      <img
                        src={sample.after}
                        alt={`${sample.name} after`}
                        className="w-full h-24 object-cover rounded-lg border border-gray-600"
                      />
                      <p className="text-xs text-center mt-2 text-gray-300 font-medium">After</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            
            {selectedSample && (
              <div className="mt-6 p-4 bg-gradient-to-r from-green-500/20 to-emerald-500/20 border border-green-500/30 rounded-xl">
                <p className="text-sm text-green-300 flex items-center justify-center">
                  <span className="w-2 h-2 bg-green-400 rounded-full mr-2 animate-pulse"></span>
                  Sample images loaded! Click "Analyze Changes with MCP" to see AI-powered results.
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="bg-gray-800/50 border-gray-700/50 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="text-white text-2xl font-bold">AI Analysis Results</CardTitle>
            <CardDescription className="text-gray-300 text-lg">
              OpenCV change detection + GPT-4 Vision insights powered by MCP
            </CardDescription>
          </CardHeader>
          <CardContent>
            {!results ? (
              <div className="text-center text-gray-400 py-12">
                <div className="text-6xl mb-4">🛰️</div>
                <p className="text-lg">Upload and analyze images to see AI-powered results</p>
                <p className="text-sm text-gray-500 mt-2">MCP tools ready for change detection</p>
              </div>
            ) : (
              <div className="space-y-6">
                {results.success ? (
                  <>
                    {/* Change Statistics */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                      <div className="text-center p-6 bg-gradient-to-br from-blue-500/20 to-blue-600/20 border border-blue-500/30 rounded-xl">
                        <div className="text-3xl font-bold text-blue-300 mb-2">
                          {results.results.change_percentage}%
                        </div>
                        <div className="text-sm text-gray-300 font-medium">Change Detected</div>
                      </div>
                      <div className="text-center p-6 bg-gradient-to-br from-green-500/20 to-green-600/20 border border-green-500/30 rounded-xl">
                        <div className="text-3xl font-bold text-green-300 mb-2">
                          {results.results.changed_pixels.toLocaleString()}
                        </div>
                        <div className="text-sm text-gray-300 font-medium">Pixels Changed</div>
                      </div>
                      <div className="text-center p-6 bg-gradient-to-br from-purple-500/20 to-purple-600/20 border border-purple-500/30 rounded-xl">
                        <div className="text-3xl font-bold text-purple-300 mb-2">
                          {results.results.total_pixels.toLocaleString()}
                        </div>
                        <div className="text-sm text-gray-300 font-medium">Total Pixels</div>
                      </div>
                      <div className="text-center p-6 bg-gradient-to-br from-orange-500/20 to-orange-600/20 border border-orange-500/30 rounded-xl">
                        <div className="text-3xl font-bold text-orange-300 mb-2">
                          {results.results.contours_count}
                        </div>
                        <div className="text-sm text-gray-300 font-medium">Change Regions</div>
                      </div>
                    </div>

                    {/* Change Mask Visualization */}
                    {results.results.change_mask_base64 && (
                      <div className="space-y-6">
                        <h3 className="text-xl font-bold text-white">🎯 AI Change Detection Overlay</h3>
                        <div className="flex justify-center bg-gray-900/50 rounded-xl p-6 border border-gray-600">
                          <img 
                            src={`data:image/png;base64,${results.results.change_mask_base64}`}
                            alt="Change detection overlay"
                            className="max-w-full h-auto border-2 border-purple-500/30 rounded-lg shadow-2xl"
                          />
                        </div>
                        <div className="text-center p-4 bg-gradient-to-r from-green-500/20 to-emerald-500/20 border border-green-500/30 rounded-lg">
                          <p className="text-sm text-green-300 flex items-center justify-center">
                            <span className="mr-2">🟢</span>
                            Green areas highlight detected changes using OpenCV algorithms
                            <span className="ml-2">🟢</span>
                          </p>
                        </div>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="text-center py-12">
                    <div className="text-6xl mb-4">⚠️</div>
                    <div className="p-6 bg-gradient-to-r from-red-500/20 to-red-600/20 border border-red-500/30 rounded-xl">
                      <p className="text-red-300 text-lg font-medium">Analysis failed</p>
                      <p className="text-red-400 text-sm mt-2">{results.error}</p>
                      <p className="text-gray-400 text-xs mt-3">MCP backend error - please try again</p>
                    </div>
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
