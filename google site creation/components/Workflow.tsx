import React from 'react';
import { Upload, ArrowRight, GitMerge, LineChart, CheckCircle2 } from 'lucide-react';

const Workflow: React.FC = () => {
  return (
    <section id="workflow" className="py-24 bg-slate-900 text-white relative overflow-hidden">
       {/* Background Pattern */}
       <div className="absolute inset-0 bg-[linear-gradient(to_right,#4f4f4f2e_1px,transparent_1px),linear-gradient(to_bottom,#4f4f4f2e_1px,transparent_1px)] bg-[size:14px_24px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)]"></div>
      
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <div className="text-center mb-20">
          <h2 className="text-3xl md:text-4xl font-bold mb-6">How it Works</h2>
          <p className="text-xl text-slate-400 max-w-3xl mx-auto">
            Your catalog is messy. Our software cleans it up. <br/>
            Here is the typical workflow for a business user.
          </p>
        </div>

        <div className="relative">
            {/* Connecting Line (Desktop) */}
            <div className="hidden md:block absolute top-1/2 left-0 w-full h-1 bg-indigo-900/50 -translate-y-1/2 z-0"></div>

            <div className="grid md:grid-cols-3 gap-12 relative z-10">
                {/* Step 1 */}
                <div className="bg-slate-800 border border-slate-700 rounded-2xl p-8 text-center transform hover:-translate-y-2 transition-transform duration-300 shadow-2xl">
                    <div className="w-16 h-16 bg-indigo-600 rounded-full flex items-center justify-center mx-auto mb-6 text-white shadow-lg shadow-indigo-600/20">
                        <Upload size={32} />
                    </div>
                    <h3 className="text-xl font-bold mb-4">1. Batch Upload</h3>
                    <p className="text-slate-400 leading-relaxed">
                        Select a "New Arrivals" folder and a "Legacy Catalog" folder from your local drive. The app ingests thousands of images in seconds.
                    </p>
                </div>

                {/* Step 2 */}
                <div className="bg-slate-800 border border-slate-700 rounded-2xl p-8 text-center transform hover:-translate-y-2 transition-transform duration-300 shadow-2xl">
                    <div className="w-16 h-16 bg-purple-600 rounded-full flex items-center justify-center mx-auto mb-6 text-white shadow-lg shadow-purple-600/20">
                        <GitMerge size={32} />
                    </div>
                    <h3 className="text-xl font-bold mb-4">2. AI Matching</h3>
                    <p className="text-slate-400 leading-relaxed">
                        The engine identifies visually similar items. It flags exact matches, suggests variants for similar items, and categorizes completely new stock.
                    </p>
                </div>

                {/* Step 3 */}
                <div className="bg-slate-800 border border-slate-700 rounded-2xl p-8 text-center transform hover:-translate-y-2 transition-transform duration-300 shadow-2xl">
                    <div className="w-16 h-16 bg-emerald-600 rounded-full flex items-center justify-center mx-auto mb-6 text-white shadow-lg shadow-emerald-600/20">
                        <LineChart size={32} />
                    </div>
                    <h3 className="text-xl font-bold mb-4">3. Analytics & Export</h3>
                    <p className="text-slate-400 leading-relaxed">
                        Review the breakdown. See price history for matched items, approve new entries, and export the clean dataset to your inventory system.
                    </p>
                </div>
            </div>
        </div>

        {/* UI Mockup / Visual */}
        <div className="mt-24 bg-slate-800 rounded-xl border border-slate-700 p-2 shadow-2xl max-w-4xl mx-auto">
            <div className="bg-slate-900 rounded-lg overflow-hidden border border-slate-800">
                <div className="flex items-center gap-2 px-4 py-3 border-b border-slate-800 bg-slate-800/50">
                    <div className="flex gap-1.5">
                        <div className="w-3 h-3 rounded-full bg-red-500"></div>
                        <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                        <div className="w-3 h-3 rounded-full bg-green-500"></div>
                    </div>
                    <div className="ml-4 text-xs text-slate-400 font-mono">CatalogSync AI - Project: Summer_Inventory_2024</div>
                </div>
                <div className="p-6 grid md:grid-cols-2 gap-6">
                    {/* Left: List */}
                    <div className="space-y-3">
                        <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">New Uploads (Processing...)</div>
                        {[1, 2, 3].map((i) => (
                            <div key={i} className="flex items-center gap-3 p-3 bg-slate-800/50 rounded border border-slate-700">
                                <div className="w-10 h-10 bg-slate-700 rounded flex items-center justify-center text-xs text-slate-500">Img</div>
                                <div className="flex-1 min-w-0">
                                    <div className="h-2 w-24 bg-slate-600 rounded mb-1.5"></div>
                                    <div className="h-2 w-16 bg-slate-700 rounded"></div>
                                </div>
                                {i === 1 ? <div className="text-emerald-400 text-xs font-bold">MATCH</div> : 
                                 i === 2 ? <div className="text-yellow-400 text-xs font-bold">VARIANT</div> :
                                 <div className="w-4 h-4 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin"></div>}
                            </div>
                        ))}
                    </div>
                    {/* Right: Details */}
                    <div className="bg-slate-800 rounded p-4 border border-slate-700">
                        <div className="flex justify-between items-center mb-4">
                            <div className="text-sm font-bold text-white">Blue Denim Jacket v2</div>
                            <div className="px-2 py-0.5 bg-emerald-500/20 text-emerald-400 text-xs rounded border border-emerald-500/30 flex items-center gap-1">
                                <CheckCircle2 size={10} />
                                Exact Match
                            </div>
                        </div>
                        <div className="space-y-4">
                            <div>
                                <div className="text-xs text-slate-500 mb-1">Matched with Legacy ID</div>
                                <div className="font-mono text-sm text-indigo-300">SKU-9928-XJ</div>
                            </div>
                            <div>
                                <div className="text-xs text-slate-500 mb-1">Price History (6 Mo)</div>
                                <div className="h-16 flex items-end gap-1">
                                    {[40, 65, 50, 80, 70, 90].map((h, idx) => (
                                        <div key={idx} className="bg-indigo-500/50 w-full rounded-t" style={{height: `${h}%`}}></div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
      </div>
    </section>
  );
};

export default Workflow;