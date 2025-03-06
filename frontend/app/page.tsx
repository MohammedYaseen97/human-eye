'use client';

import { useState, useRef, useEffect } from 'react';
import Image from 'next/image';

// Add this type definition
type PlatformType = 'android' | 'ios' | 'desktop';

export default function Home() {
  const [image, setImage] = useState<string | null>(null);
  const [streamImage, setStreamImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [params, setParams] = useState({
    age: 25,
    platform: 'android' as PlatformType,
    task: 'find settings',
    techSaviness: 3,
  });
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [currentRequest, setCurrentRequest] = useState<AbortController | null>(null);

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
    setStreamImage(null); // Clear previous image

    try {
      const formData = new FormData();
      const imageBlob = await fetch(image).then(r => r.blob());
      formData.append('file', imageBlob, 'image.png');
      formData.append('age', params.age.toString());
      formData.append('platform', params.platform);
      formData.append('task', params.task);
      formData.append('tech_saviness', params.techSaviness.toString());

      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
        signal: abortController.signal
      });

      if (!response.ok) throw new Error('Prediction failed');

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No reader available');

      // Buffer to hold partial chunks
      let buffer = '';

      // Read the stream
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // Convert chunk to text and add to buffer
        const text = new TextDecoder().decode(value);
        buffer += text;

        // Process complete messages from buffer
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || ''; // Keep the last incomplete chunk in buffer

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const jsonData = JSON.parse(line.slice(6));
              
              if (jsonData.status === 'error') {
                setError(jsonData.message);
                break;
              }

              // Use requestAnimationFrame for smooth updates
              if (jsonData.timestep) {
                requestAnimationFrame(() => {
                  setStreamImage(jsonData.timestep);
                });
              }
            } catch (e) {
              console.log('Error parsing JSON from line:', line);
            }
          }
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        console.log('Request was cancelled');
      } else {
        setError(err instanceof Error ? err.message : 'An error occurred');
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
  }, []);

  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-white p-8">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-4xl font-bold mb-8 text-center bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500">
          UI Attention Predictor
        </h1>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Left Column - Input */}
          <div className="space-y-6 bg-gray-800 p-6 rounded-xl shadow-xl">
            <div className="space-y-4">
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
                  <label className="block text-sm font-medium mb-2">Platform</label>
                  <select
                    value={params.platform}
                    onChange={(e) => setParams({ ...params, platform: e.target.value as PlatformType })}
                    className="w-full bg-gray-700 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="android">Android</option>
                    <option value="ios">iOS</option>
                    <option value="desktop">Desktop</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Task</label>
                <input
                  type="text"
                  value={params.task}
                  onChange={(e) => setParams({ ...params, task: e.target.value })}
                  className="w-full bg-gray-700 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500"
                  placeholder="e.g., find settings"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Tech Savviness (1-10)</label>
                <input
                  type="range"
                  min="1"
                  max="10"
                  value={params.techSaviness}
                  onChange={(e) => setParams({ ...params, techSaviness: parseInt(e.target.value) })}
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                />
                <div className="text-center mt-2">{params.techSaviness}</div>
              </div>
            </div>

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

          {/* Right Column - Output */}
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
        </div>
      </div>
    </main>
  );
}
