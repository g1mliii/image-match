import React, { useState, useEffect } from 'react';
import { Layers, Menu, X } from 'lucide-react';

const Navbar: React.FC = () => {
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToSection = (id: string) => {
    setIsMobileMenuOpen(false);
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    } else if (id === 'root') {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  const navLinks = [
    { name: 'Features', id: 'features' },
    { name: 'How it Works', id: 'workflow' },
    { name: 'Pricing', id: 'pricing' },
  ];

  return (
    <nav 
      className={`fixed top-0 w-full z-50 transition-all duration-300 ${
        isScrolled || isMobileMenuOpen ? 'bg-white/90 backdrop-blur-md shadow-sm border-b border-slate-200' : 'bg-transparent'
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center">
            <button 
              onClick={() => scrollToSection('root')}
              className="flex items-center space-x-2 group"
            >
              <div className="p-2 bg-indigo-600 rounded-lg group-hover:bg-indigo-700 transition-colors">
                <Layers className="h-6 w-6 text-white" />
              </div>
              <span className="text-xl font-bold text-slate-900">CatalogSync AI</span>
            </button>
          </div>
          
          {/* Desktop Nav */}
          <div className="hidden md:flex items-center space-x-8">
            {navLinks.map((link) => (
              <button 
                key={link.name} 
                onClick={() => scrollToSection(link.id)}
                className="text-sm font-medium text-slate-600 hover:text-indigo-600 transition-colors cursor-pointer bg-transparent border-none"
              >
                {link.name}
              </button>
            ))}
            <button 
              className="px-4 py-2 text-sm font-medium text-white bg-slate-900 rounded-full hover:bg-slate-800 transition-colors shadow-lg shadow-slate-900/20"
            >
              Download Beta
            </button>
          </div>

          {/* Mobile Toggle */}
          <div className="md:hidden">
            <button 
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="p-2 text-slate-600 hover:text-slate-900"
            >
              {isMobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Menu */}
      {isMobileMenuOpen && (
        <div className="md:hidden bg-white border-t border-slate-100">
          <div className="px-4 pt-2 pb-6 space-y-1">
            {navLinks.map((link) => (
              <button
                key={link.name}
                onClick={() => scrollToSection(link.id)}
                className="block w-full text-left px-3 py-4 text-base font-medium text-slate-600 hover:text-indigo-600 hover:bg-slate-50 rounded-md"
              >
                {link.name}
              </button>
            ))}
            <div className="pt-4">
              <button 
                className="block w-full text-center px-4 py-3 text-base font-medium text-white bg-indigo-600 rounded-lg hover:bg-indigo-700"
              >
                Download Beta
              </button>
            </div>
          </div>
        </div>
      )}
    </nav>
  );
};

export default Navbar;