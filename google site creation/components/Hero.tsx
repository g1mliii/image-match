import React from 'react';
import { ArrowRight, Database, RefreshCw, TrendingUp, Monitor } from 'lucide-react';

const Hero: React.FC = () => {
  return (
    <div className="relative pt-32 pb-20 lg:pt-40 lg:pb-28 overflow-hidden">
      {/* Abstract Background Shapes */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full h-full -z-10">
        <div className="absolute top-20 left-10 w-72 h-72 bg-indigo-200 rounded-full mix-blend-multiply filter blur-3xl opacity-30 animate-pulse"></div>
        <div className="absolute top-20 right-10 w-72 h-72 bg-purple-200 rounded-full mix-blend-multiply filter blur-3xl opacity-30 animate-pulse delay-75"></div>
        <div className="absolute -bottom-8 left-1/2 w-96 h-96 bg-teal-200 rounded-full mix-blend-multiply filter blur-3xl opacity-30 animate-pulse delay-150"></div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative">
        <div className="text-center max-w-4xl mx-auto">
          <div className="inline-flex items-center px-3 py-1 rounded-full bg-indigo-50 border border-indigo-100 text-indigo-600 text-sm font-medium mb-6">
            <span className="flex h-2 w-2 bg-indigo-600 rounded-full mr-2"></span>
            Desktop Software v1.0
          </div>
          
          <h1 className="text-5xl md:text-7xl font-bold text-slate-900 tracking-tight mb-8">
            Match your inventory <br/>
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 to-purple-600">
              in bulk with AI.
            </span>
          </h1>
          
          <p className="text-xl text-slate-600 mb-10 max-w-2xl mx-auto leading-relaxed">
            The comprehensive desktop tool for businesses. Drag and drop thousands of new product images to automatically match them against your legacy catalog.
          </p>

          <div className="flex flex-col sm:flex-row justify-center gap-4">
            <button className="px-8 py-4 text-lg font-semibold text-white bg-indigo-600 rounded-xl hover:bg-indigo-700 transition-all shadow-xl shadow-indigo-200 flex items-center justify-center">
              Download for Windows
              <ArrowRight className="ml-2 h-5 w-5" />
            </button>
            <button className="px-8 py-4 text-lg font-semibold text-slate-700 bg-white border border-slate-200 rounded-xl hover:bg-slate-50 transition-all flex items-center justify-center">
              View Documentation
            </button>
          </div>

          {/* Stats/Trust */}
          <div className="mt-16 pt-8 border-t border-slate-200 grid grid-cols-2 md:grid-cols-4 gap-8">
            <div className="flex flex-col items-center">
               <Database className="h-6 w-6 text-slate-400 mb-2" />
               <span className="font-bold text-2xl text-slate-900">Local</span>
               <span className="text-sm text-slate-500">Database Processing</span>
            </div>
            <div className="flex flex-col items-center">
               <RefreshCw className="h-6 w-6 text-slate-400 mb-2" />
               <span className="font-bold text-2xl text-slate-900">1000+</span>
               <span className="text-sm text-slate-500">Images per Batch</span>
            </div>
            <div className="flex flex-col items-center">
               <TrendingUp className="h-6 w-6 text-slate-400 mb-2" />
               <span className="font-bold text-2xl text-slate-900">6-Month</span>
               <span className="text-sm text-slate-500">Price History</span>
            </div>
             <div className="flex flex-col items-center">
               <Monitor className="h-6 w-6 text-slate-400 mb-2" />
               <span className="font-bold text-2xl text-slate-900">macOS/Win</span>
               <span className="text-sm text-slate-500">Native Support</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Hero;