import React, { useState, useRef } from 'react';
import { Upload, Loader2, CheckCircle2, AlertCircle, Tag, DollarSign, BarChart3 } from 'lucide-react';
import { analyzeProductImage } from '../services/geminiService';
import { AnalysisResult } from '../types';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const DemoSection: React.FC = () => {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelection(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelection = async (file: File) => {
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file.');
      return;
    }
    
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setResult(null);
    
    setIsAnalyzing(true);
    try {
      const analysis = await analyzeProductImage(file);
      setResult(analysis);
    } catch (error) {
      console.error(error);
      alert("Failed to analyze image. Please try again.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <section id="demo" className="py-24 bg-slate-900 text-white relative overflow-hidden">
      {/* Background Grid */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#4f4f4f2e_1px,transparent_1px),linear-gradient(to_bottom,#4f4f4f2e_1px,transparent_1px)] bg-[size:14px_24px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)]"></div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">See the Engine in Action</h2>
          <p className="text-slate-400 max-w-2xl mx-auto">
            Upload any product image to test our AI categorization and matching engine. 
            It will simulate a match against a global database and provide price analytics.
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-12 items-start">
          {/* Upload Area */}
          <div className="space-y-6">
            <div 
              className={`border-2 border-dashed rounded-2xl p-8 transition-all duration-300 flex flex-col items-center justify-center h-96 ${
                isDragging 
                  ? 'border-indigo-500 bg-indigo-500/10' 
                  : 'border-slate-700 bg-slate-800/50 hover:border-slate-600'
              }`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              {previewUrl ? (
                <div className="relative w-full h-full flex flex-col items-center justify-center">
                   <img 
                    src={previewUrl} 
                    alt="Preview" 
                    className="max-h-64 max-w-full rounded-lg shadow-lg object-contain mb-4" 
                  />
                   <button 
                    onClick={() => {
                      setPreviewUrl(null);
                      setResult(null);
                      setSelectedFile(null);
                    }}
                    className="text-sm text-slate-400 hover:text-white underline"
                  >
                    Upload different image
                  </button>
                </div>
              ) : (
                <>
                  <div className="h-16 w-16 bg-indigo-600/20 text-indigo-400 rounded-full flex items-center justify-center mb-4">
                    <Upload size={32} />
                  </div>
                  <h3 className="text-xl font-semibold mb-2">Drop product image here</h3>
                  <p className="text-slate-400 text-sm mb-6">or click to browse (JPG, PNG)</p>
                  <button 
                    onClick={() => fileInputRef.current?.click()}
                    className="px-6 py-3 bg-indigo-600 hover:bg-indigo-700 rounded-lg font-medium transition-colors"
                  >
                    Select File
                  </button>
                </>
              )}
              <input 
                type="file" 
                ref={fileInputRef}
                className="hidden"
                accept="image/*"
                onChange={(e) => e.target.files?.[0] && handleFileSelection(e.target.files[0])}
              />
            </div>
            
            <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
                <div className="flex items-center gap-3 mb-2">
                    <AlertCircle className="text-indigo-400 w-5 h-5" />
                    <p className="text-sm font-semibold text-indigo-100">How it works</p>
                </div>
                <p className="text-xs text-slate-400">
                    The uploaded image is analyzed by our Gemini-powered vision model. 
                    Features are extracted and compared against a vector index of your inventory (simulated here) to find exact matches or variants.
                </p>
            </div>
          </div>

          {/* Results Area */}
          <div className="min-h-[400px]">
            {isAnalyzing && (
              <div className="h-full flex flex-col items-center justify-center space-y-4 animate-pulse">
                <Loader2 className="w-12 h-12 text-indigo-500 animate-spin" />
                <p className="text-slate-400 font-medium">Analyzing visual features...</p>
                <p className="text-slate-500 text-sm">Comparing against inventory database</p>
              </div>
            )}

            {!isAnalyzing && !result && (
              <div className="h-full flex flex-col items-center justify-center border border-slate-800 rounded-2xl bg-slate-900/50">
                 <div className="text-slate-600 text-center p-8">
                    <BarChart3 className="w-16 h-16 mx-auto mb-4 opacity-20" />
                    <p className="text-lg font-medium">Results will appear here</p>
                    <p className="text-sm">Upload an image to see the analysis</p>
                 </div>
              </div>
            )}

            {!isAnalyzing && result && (
              <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-700">
                
                {/* Header Card */}
                <div className="bg-slate-800 rounded-2xl p-6 border border-slate-700 shadow-xl">
                    <div className="flex justify-between items-start mb-4">
                        <div>
                            <h3 className="text-2xl font-bold text-white">{result.productName}</h3>
                            <div className="flex items-center gap-2 mt-1 text-slate-400">
                                <Tag className="w-4 h-4" />
                                <span>{result.category}</span>
                            </div>
                        </div>
                        <div className={`px-3 py-1 rounded-full text-sm font-semibold border ${
                            result.matchStatus === 'Exact Match' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' :
                            result.matchStatus === 'Similar Variant' ? 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20' :
                            'bg-blue-500/10 text-blue-400 border-blue-500/20'
                        }`}>
                            {result.matchStatus}
                        </div>
                    </div>

                    <div className="grid grid-cols-3 gap-4 mt-6">
                         <div className="bg-slate-900/50 p-3 rounded-lg border border-slate-700/50">
                             <p className="text-xs text-slate-400 uppercase tracking-wider font-semibold mb-1">Confidence</p>
                             <div className="flex items-center gap-2">
                                 <CheckCircle2 className={`w-5 h-5 ${result.confidenceScore > 80 ? 'text-emerald-500' : 'text-yellow-500'}`} />
                                 <span className="text-xl font-bold">{result.confidenceScore}%</span>
                             </div>
                         </div>
                         <div className="bg-slate-900/50 p-3 rounded-lg border border-slate-700/50">
                             <p className="text-xs text-slate-400 uppercase tracking-wider font-semibold mb-1">Avg. Price</p>
                             <div className="flex items-center gap-1">
                                 <DollarSign className="w-5 h-5 text-slate-300" />
                                 <span className="text-xl font-bold">{result.suggestedPrice}</span>
                             </div>
                         </div>
                         <div className="bg-slate-900/50 p-3 rounded-lg border border-slate-700/50">
                             <p className="text-xs text-slate-400 uppercase tracking-wider font-semibold mb-1">Action</p>
                             <span className="text-sm font-medium text-indigo-300">{result.inventoryAction}</span>
                         </div>
                    </div>
                </div>

                {/* Chart Card */}
                <div className="bg-slate-800 rounded-2xl p-6 border border-slate-700 shadow-xl">
                    <h4 className="text-lg font-semibold mb-6 flex items-center gap-2">
                        <TrendingUpIcon />
                        Price History Trend
                    </h4>
                    <div className="h-48 w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={result.priceHistory}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                                <XAxis 
                                    dataKey="month" 
                                    stroke="#94a3b8" 
                                    tick={{fontSize: 12}} 
                                    tickLine={false}
                                    axisLine={false}
                                />
                                <YAxis 
                                    stroke="#94a3b8" 
                                    tick={{fontSize: 12}} 
                                    tickLine={false}
                                    axisLine={false}
                                    tickFormatter={(value) => `$${value}`}
                                />
                                <Tooltip 
                                    contentStyle={{backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'}}
                                    itemStyle={{color: '#fff'}}
                                />
                                <Line 
                                    type="monotone" 
                                    dataKey="price" 
                                    stroke="#6366f1" 
                                    strokeWidth={3} 
                                    dot={{fill: '#6366f1', strokeWidth: 2}} 
                                    activeDot={{r: 6, fill: '#fff'}}
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>
                
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
};

// Simple Icon Helper for inside component
const TrendingUpIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-indigo-400"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline></svg>
)

export default DemoSection;