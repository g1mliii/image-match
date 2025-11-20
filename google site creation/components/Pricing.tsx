import React from 'react';
import { Check, X } from 'lucide-react';
import { PricingTier } from '../types';

const Pricing: React.FC = () => {
  const tiers: PricingTier[] = [
    {
      name: "Starter",
      price: "Free",
      description: "For hobbyists or testing the engine.",
      features: [
        "Manage up to 100 Products",
        "Basic Image Matching",
        "Manual Categorization",
        "Local Storage Only",
        "Community Support"
      ],
      cta: "Download Free",
      highlight: false
    },
    {
      name: "Business",
      price: "$49",
      description: "For serious sellers with large inventories.",
      features: [
        "Manage up to 5,000 Products",
        "Unlimited Folder Uploads",
        "AI Auto-Categorization",
        "Full Price History Analytics",
        "Export to CSV/Excel",
        "Priority Email Support"
      ],
      cta: "Buy License",
      highlight: true
    }
  ];

  return (
    <section id="pricing" className="py-24 bg-slate-50 border-t border-slate-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-slate-900 mb-4">Software Licenses</h2>
          <p className="text-xl text-slate-600">One-time setup, simple monthly licensing for cloud features.</p>
        </div>

        <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
          {tiers.map((tier) => (
            <div 
              key={tier.name}
              className={`relative rounded-3xl p-8 xl:p-10 transition-all duration-300 ${
                tier.highlight 
                  ? 'bg-white shadow-2xl border-2 border-indigo-600 scale-105 z-10' 
                  : 'bg-white shadow-lg border border-slate-200 hover:border-indigo-300'
              }`}
            >
              {tier.highlight && (
                <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2">
                  <span className="bg-indigo-600 text-white px-4 py-1 rounded-full text-sm font-semibold uppercase tracking-wide shadow-md">
                    Recommended
                  </span>
                </div>
              )}

              <h3 className="text-2xl font-bold text-slate-900">{tier.name}</h3>
              <div className="mt-4 flex items-baseline">
                <span className="text-5xl font-extrabold tracking-tight text-slate-900">{tier.price}</span>
                {tier.price !== "Free" && <span className="ml-2 text-xl text-slate-500 font-medium">/month</span>}
              </div>
              <p className="mt-4 text-slate-500">{tier.description}</p>

              <ul className="mt-8 space-y-4">
                {tier.features.map((feature) => (
                  <li key={feature} className="flex items-center">
                    <div className="flex-shrink-0">
                      <Check className="h-5 w-5 text-indigo-600" />
                    </div>
                    <p className="ml-3 text-base text-slate-600">{feature}</p>
                  </li>
                ))}
                 {!tier.highlight && (
                    <>
                        <li className="flex items-center opacity-50">
                            <div className="flex-shrink-0">
                                <X className="h-5 w-5 text-slate-400" />
                            </div>
                            <p className="ml-3 text-base text-slate-500">AI Auto-Categorization</p>
                        </li>
                        <li className="flex items-center opacity-50">
                            <div className="flex-shrink-0">
                                <X className="h-5 w-5 text-slate-400" />
                            </div>
                            <p className="ml-3 text-base text-slate-500">CSV Export</p>
                        </li>
                    </>
                 )}
              </ul>

              <div className="mt-10">
                <button 
                  className={`w-full block text-center rounded-xl px-6 py-4 text-lg font-bold transition-all ${
                    tier.highlight
                      ? 'bg-indigo-600 text-white hover:bg-indigo-700 shadow-lg hover:shadow-indigo-200'
                      : 'bg-slate-100 text-slate-900 hover:bg-slate-200'
                  }`}
                >
                  {tier.cta}
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Pricing;