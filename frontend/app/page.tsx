'use client';

import { useState, useRef, useEffect } from 'react';
import Image from 'next/image';
import AttentionPoints from './components/AttentionPoints';
import type { AttentionPoint } from './components/AttentionPoints';

export default function Home() {
  const [image, setImage] = useState<string | null>(null);
  const [streamImage, setStreamImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [params, setParams] = useState({
    age: 0,
    task: '',
    techSaviness: 5,
    platform: 'desktop' as 'desktop' | 'ios' | 'android',
  });
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [currentRequest, setCurrentRequest] = useState<AbortController | null>(null);
  const [attentionPoints, setAttentionPoints] = useState<AttentionPoint[]>([]);
  const [selectedPoint, setSelectedPoint] = useState<AttentionPoint | null>(null);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImage(reader.result as string);
        setStreamImage(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = async () => {
    if (!image) return;

    // Cancel any existing request
    if (currentRequest) {
      currentRequest.abort();
    }

    // Create new abort controller for this request
    const abortController = new AbortController();
    setCurrentRequest(abortController);
    setLoading(true);
    setError(null);
    setStreamImage(null);

    try {
      // Prepare form data
      const formData = new FormData();
      // Ensure image exists (TypeScript check)
      if (!image) return;
      const imageBlob = await fetch(image).then(r => r.blob());
      formData.append('file', imageBlob, 'image.png');
      formData.append('age', params.age.toString());
      formData.append('task', params.task);
      formData.append('tech_saviness', params.techSaviness.toString());
      formData.append('platform', params.platform.toLowerCase());

      // Make the request
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      console.log('Making request to:', `${apiUrl}/predict`);
      
      const response = await fetch(`${apiUrl}/predict`, {
        method: 'POST',
        body: formData,
        headers: {
          'Accept': 'application/json',
        },
        signal: abortController.signal
      });

      // Handle response
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Prediction failed: ${errorText}`);
      }

      // Parse and handle the result
      const result = await response.json();
      if (result.timestep) {
        setStreamImage(result.timestep);
      }

      // Update attention points from the response
      if (result && result.elements) {
        setAttentionPoints(result.elements);
      }

    } catch (err) {
      const error = err as Error;
      if (error.name === 'AbortError') {
        console.log('Request was cancelled');
      } else {
        setError(error.message || 'An error occurred');
      }
    } finally {
      setLoading(false);
      setCurrentRequest(null);
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (currentRequest) {
        currentRequest.abort();
      }
    };
  }, [currentRequest]);

  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-white p-8">
      <div className="max-w-[1600px] mx-auto">
        <h1 className="text-4xl font-bold mb-8 text-center bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500">
          UI Attention Predictor
        </h1>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column - Input */}
          <div className="space-y-6 bg-gray-800 p-6 rounded-xl shadow-xl">
            {/* Image Upload */}
            <div>
              <label className="block text-sm font-medium mb-2">Upload UI Screenshot</label>
              <div 
                className="border-2 border-dashed border-gray-600 rounded-lg p-8 text-center cursor-pointer hover:border-blue-500 transition-colors"
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  type="file"
                  ref={fileInputRef}
                  className="hidden"
                  accept="image/*"
                  onChange={handleImageUpload}
                />
                {image ? (
                  <Image
                    src={image}
                    alt="Preview"
                    width={300}
                    height={300}
                    className="mx-auto rounded-lg"
                  />
                ) : (
                  <div className="text-gray-400">
                    <svg className="mx-auto h-12 w-12 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    <p>Click to upload an image</p>
                  </div>
                )}
              </div>
            </div>

            {/* Parameters Section */}
            <div className="space-y-4">
              {/* Task Input */}
              <div>
                <label className="block text-sm font-medium mb-2">Task Description</label>
                <input
                  type="text"
                  value={params.task}
                  onChange={(e) => setParams({ ...params, task: e.target.value })}
                  className="w-full bg-gray-700 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500"
                  placeholder="e.g., find settings"
                />
              </div>

              {/* Platform Selection */}
              <div>
                <label className="block text-sm font-medium mb-2">Platform</label>
                <select
                  value={params.platform}
                  onChange={(e) => setParams({ ...params, platform: e.target.value as 'desktop' | 'ios' | 'android' })}
                  className="w-full bg-gray-700 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500"
                >
                  <option value="desktop">Desktop</option>
                  <option value="ios">iOS</option>
                  <option value="android">Android</option>
                </select>
              </div>

              {/* Age and Tech Savviness in a row */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Age</label>
                  <input
                    type="number"
                    value={params.age}
                    onChange={(e) => setParams({ ...params, age: parseInt(e.target.value) })}
                    className="w-full bg-gray-700 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">Tech Savviness</label>
                  <div className="space-y-2">
                    <input
                      type="range"
                      min="1"
                      max="10"
                      value={params.techSaviness}
                      onChange={(e) => setParams({ ...params, techSaviness: parseInt(e.target.value) })}
                      className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                    />
                    <div className="text-center text-sm">{params.techSaviness}/10</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Submit Button */}
            <button
              onClick={handleSubmit}
              disabled={!image || loading}
              className={`w-full py-3 px-4 rounded-lg font-medium transition-all ${
                loading || !image
                  ? 'bg-gray-600 cursor-not-allowed'
                  : 'bg-blue-500 hover:bg-blue-600'
              }`}
            >
              {loading ? (
                <span className="flex items-center justify-center">
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Processing...
                </span>
              ) : 'Start Prediction'}
            </button>
          </div>

          {/* Middle Column - Output Visualization */}
          <div className="space-y-6 bg-gray-800 p-6 rounded-xl shadow-xl">
            <h2 className="text-xl font-semibold mb-4">Attention Visualization</h2>
            <div className="aspect-square relative bg-gray-700 rounded-lg overflow-hidden">
              {streamImage ? (
                <Image
                  src={streamImage}
                  alt="Attention Visualization"
                  fill
                  className="object-contain transition-opacity duration-200"
                  priority
                />
              ) : (
                <div className="absolute inset-0 flex items-center justify-center text-gray-400">
                  {loading ? (
                    <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
                  ) : (
                    'Upload an image and start prediction'
                  )}
                </div>
              )}
            </div>
            {error && (
              <div className="bg-red-500/10 border border-red-500 text-red-500 px-4 py-2 rounded-lg">
                {error}
              </div>
            )}
          </div>

          {/* Right Column - Attention Points Analysis */}
          <div className="h-full">
            {attentionPoints.length > 0 && (
              <AttentionPoints
                points={attentionPoints}
                selectedPoint={selectedPoint}
                onPointHover={(point) => {
                  setSelectedPoint(point);
                }}
                onPointClick={(point) => {
                  setSelectedPoint(point);
                }}
              />
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
