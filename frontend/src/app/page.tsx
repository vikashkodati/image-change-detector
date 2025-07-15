"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export default function Home() {
  const [beforeImage, setBeforeImage] = useState<File | null>(null);
  const [afterImage, setAfterImage] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState<any>(null);

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
      
      // Show basic results for now
      if (data.success) {
        const changePercent = data.results.change_percentage;
        alert(`Analysis complete! ${changePercent}% of pixels changed.`);
      } else {
        alert(`Analysis failed: ${data.error}`);
      }
      
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

        <Card>
          <CardHeader>
            <CardTitle>Analysis Results</CardTitle>
            <CardDescription>
              Change detection results will appear here
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-center text-gray-500 py-8">
              <p>Upload and analyze images to see results</p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
