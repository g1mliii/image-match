import React from 'react';
import { Box, Search, Database, Zap, FolderInput, BarChart } from 'lucide-react';

const Features: React.FC = () => {
  const features = [
    {
      icon: <FolderInput className="w-6 h-6 text-indigo-600" />,
      title: "Batch Folder Processing",
      description: "Don't upload one by one. Drag entire folders containing thousands of product images into the app for immediate local processing."
    },
    {
      icon: <Search className="w-6 h-6 text-indigo-600" />,
      title: "Legacy Catalog Matching",
      description: "The engine visually scans your new uploads and cross-references them with your 'Legacy' inventory folder to find duplicates and variants."
    },
    {
      icon: <BarChart className="w-6 h-6 text-indigo-600" />,
      title: "Price History & Analytics",
      description: "Matched products automatically pull generated analytics, including 6-month price history curves and stock level predictions."
    },
    {
      icon: <Database className="w-6 h-6 text-indigo-600" />,
      title: "Smart Categorization",
      description: "Define your business rules once. The AI sorts new inventory into your specific categories based on visual traits and metadata."
    },
    {
      icon: <Zap className="w-6 h-6 text-indigo-600" />,
      title: "Instant Status Reporting",
      description: "Get immediate feedback: 'Exact Match', 'New Item', or 'Variant Found' for every single image in your batch."
    },
    {
      icon: <Box className="w-6 h-6 text-indigo-600" />,
      title: "Export & Integration",
      description: "Once matched, export your consolidated inventory data to CSV, Excel, or directly to your ERP system via API (Pro only)."
    }
  ];

  return (
    <section id="features" className="py-20 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center max-w-3xl mx-auto mb-16">
          <h2 className="text-base font-semibold text-indigo-600 tracking-wide uppercase">Powerful Features</h2>
          <p className="mt-2 text-3xl leading-8 font-extrabold text-slate-900 sm:text-4xl">
            Built for High-Volume Inventory
          </p>
          <p className="mt-4 max-w-2xl text-xl text-slate-500 mx-auto">
            Stop manually checking spreadsheets. Let our vision engine match your new stock to your old catalog in seconds.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <div 
              key={index}
              className="relative group bg-slate-50 p-8 rounded-2xl border border-slate-100 hover:border-indigo-100 hover:shadow-xl hover:shadow-indigo-100/50 transition-all duration-300"
            >
              <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-indigo-500 to-purple-500 transform scale-x-0 group-hover:scale-x-100 transition-transform duration-300 rounded-t-2xl origin-left"></div>
              
              <div className="w-12 h-12 bg-white rounded-xl border border-slate-200 shadow-sm flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                {feature.icon}
              </div>
              
              <h3 className="text-xl font-bold text-slate-900 mb-3">
                {feature.title}
              </h3>
              
              <p className="text-slate-600 leading-relaxed">
                {feature.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Features;