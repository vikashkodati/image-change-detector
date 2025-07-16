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
      
      // Call FastMCP server to detect changes
      const response = await fetch('http://127.0.0.1:8000/mcp/call', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          method: 'detect_image_changes',
          params: {
            before_image_base64: beforeBase64,
            after_image_base64: afterBase64
          }
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      setResults(data);
      
    } catch (error) {
      console.error('Error analyzing images:', error);
      alert('Failed to analyze images. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Image Change Detector
          </h1>
          <p className="text-xl text-gray-600">
            Upload satellite images to detect and analyze changes
          </p>
        </div>

        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Upload Images</CardTitle>
            <CardDescription>
              Select before and after satellite images to compare
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-2">
                <Label htmlFor="before-image">Before Image</Label>
                <Input
                  id="before-image"
                  type="file"
                  accept="image/*,.tiff,.tif,.geotiff"
                  onChange={(e) => handleFileUpload(e, 'before')}
                />
                {beforeImage && (
                  <p className="text-sm text-green-600">
                    ✓ {beforeImage.name} uploaded
                  </p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="after-image">After Image</Label>
                <Input
                  id="after-image"
                  type="file"
                  accept="image/*,.tiff,.tif,.geotiff"
                  onChange={(e) => handleFileUpload(e, 'after')}
                />
                {afterImage && (
                  <p className="text-sm text-green-600">
                    ✓ {afterImage.name} uploaded
                  </p>
                )}
              </div>
            </div>

            <div className="mt-6 text-center">
              <Button 
                onClick={handleAnalyze}
                disabled={!beforeImage || !afterImage || isProcessing}
                className="px-8 py-2"
              >
                {isProcessing ? 'Analyzing...' : 'Analyze Changes'}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Sample Images Section */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Sample Disaster Imagery</CardTitle>
            <CardDescription>
              Try the app with high-resolution satellite images from real disasters
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {sampleImages.map((sample) => (
                <div
                  key={sample.id}
                  className={`border rounded-lg p-4 cursor-pointer transition-all ${
                    selectedSample === sample.id
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                  onClick={() => handleSampleSelect(sample)}
                >
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="font-semibold text-sm">{sample.name}</h3>
                    {selectedSample === sample.id && (
                      <span className="text-xs bg-blue-500 text-white px-2 py-1 rounded">
                        Selected
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-gray-600 mb-2">{sample.description}</p>
                  <p className="text-xs text-gray-500">{sample.resolution}</p>
                  
                  {/* Image preview */}
                  <div className="mt-3 flex space-x-2">
                    <div className="flex-1">
                      <img
                        src={sample.before}
                        alt={`${sample.name} before`}
                        className="w-full h-20 object-cover rounded border"
                      />
                      <p className="text-xs text-center mt-1">Before</p>
                    </div>
                    <div className="flex-1">
                      <img
                        src={sample.after}
                        alt={`${sample.name} after`}
                        className="w-full h-20 object-cover rounded border"
                      />
                      <p className="text-xs text-center mt-1">After</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            
            {selectedSample && (
              <div className="mt-4 p-3 bg-green-50 rounded-lg">
                <p className="text-sm text-green-800">
                  ✓ Sample images loaded! Click "Analyze Changes" to see the results.
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Analysis Results</CardTitle>
            <CardDescription>
              Change detection results will appear here
            </CardDescription>
          </CardHeader>
          <CardContent>
            {!results ? (
              <div className="text-center text-gray-500 py-8">
                <p>Upload and analyze images to see results</p>
              </div>
            ) : (
              <div className="space-y-6">
                {results.success ? (
                  <>
                    {/* Change Statistics */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="text-center p-4 bg-blue-50 rounded-lg">
                        <div className="text-2xl font-bold text-blue-600">
                          {results.results.change_percentage}%
                        </div>
                        <div className="text-sm text-gray-600">Change Detected</div>
                      </div>
                      <div className="text-center p-4 bg-green-50 rounded-lg">
                        <div className="text-2xl font-bold text-green-600">
                          {results.results.changed_pixels.toLocaleString()}
                        </div>
                        <div className="text-sm text-gray-600">Pixels Changed</div>
                      </div>
                      <div className="text-center p-4 bg-purple-50 rounded-lg">
                        <div className="text-2xl font-bold text-purple-600">
                          {results.results.total_pixels.toLocaleString()}
                        </div>
                        <div className="text-sm text-gray-600">Total Pixels</div>
                      </div>
                      <div className="text-center p-4 bg-orange-50 rounded-lg">
                        <div className="text-2xl font-bold text-orange-600">
                          {results.results.contours_count}
                        </div>
                        <div className="text-sm text-gray-600">Change Regions</div>
                      </div>
                    </div>

                    {/* Change Mask Visualization */}
                    {results.results.change_mask_base64 && (
                      <div className="space-y-4">
                        <h3 className="text-lg font-semibold">Change Detection Overlay</h3>
                        <div className="flex justify-center">
                          <img 
                            src={`data:image/png;base64,${results.results.change_mask_base64}`}
                            alt="Change detection overlay"
                            className="max-w-full h-auto border rounded-lg shadow-sm"
                          />
                        </div>
                        <p className="text-sm text-gray-600 text-center">
                          Green areas highlight detected changes
                        </p>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="text-center text-red-500 py-8">
                    <p>Analysis failed: {results.error}</p>
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
