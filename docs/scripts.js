// Product Matcher Website Scripts

// Lazy Loading Images
document.addEventListener('DOMContentLoaded', function() {
  // Lazy load images
  const lazyImages = document.querySelectorAll('img[data-src]');
  
  if ('IntersectionObserver' in window) {
    const imageObserver = new IntersectionObserver((entries, observer) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const img = entry.target;
          img.src = img.dataset.src;
          img.classList.add('loaded');
          img.removeAttribute('data-src');
          observer.unobserve(img);
        }
      });
    });
    
    lazyImages.forEach(img => imageObserver.observe(img));
  } else {
    // Fallback for browsers without IntersectionObserver
    lazyImages.forEach(img => {
      img.src = img.dataset.src;
      img.classList.add('loaded');
    });
  }
  
  // Smooth scroll for anchor links
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
      const href = this.getAttribute('href');
      if (href !== '#' && href !== '') {
        e.preventDefault();
        const target = document.querySelector(href);
        if (target) {
          target.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
          });
        }
      }
    });
  });
  
  // Add fade-in animation to elements as they come into view
  const fadeElements = document.querySelectorAll('.card, .step, .pricing-card');
  
  if ('IntersectionObserver' in window) {
    const fadeObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('fade-in-up');
          fadeObserver.unobserve(entry.target);
        }
      });
    }, {
      threshold: 0.1
    });
    
    fadeElements.forEach(el => fadeObserver.observe(el));
  }
  
  // Mobile menu toggle (if needed in future)
  const navbarToggle = document.querySelector('.navbar-toggle');
  const navbarNav = document.querySelector('.navbar-nav');
  
  if (navbarToggle && navbarNav) {
    navbarToggle.addEventListener('click', () => {
      navbarNav.classList.toggle('active');
    });
  }
  
  // Track download button clicks (for analytics)
  const downloadButtons = document.querySelectorAll('a[href*="download"], a[href*=".exe"], a[href*=".dmg"]');
  downloadButtons.forEach(button => {
    button.addEventListener('click', function() {
      // XSS Protection: Use textContent instead of innerHTML
      const buttonText = this.textContent || '';
      const platform = buttonText.includes('Windows') ? 'Windows' : 
                      buttonText.includes('macOS') ? 'macOS' : 'Unknown';
      console.log(`Download clicked: ${platform}`);
      // In production, send to analytics service
      // gtag('event', 'download', { platform: platform });
    });
  });
  
  // Track pricing button clicks
  const pricingButtons = document.querySelectorAll('a[href*="lemonsqueezy"]');
  pricingButtons.forEach(button => {
    button.addEventListener('click', function() {
      console.log('Pro license purchase initiated');
      // In production, send to analytics service
      // gtag('event', 'purchase_intent', { product: 'Pro License' });
    });
  });
  
  // Add active state to current page in navigation
  const currentPage = window.location.pathname.split('/').pop() || 'index.html';
  const navLinks = document.querySelectorAll('.navbar-nav a');
  navLinks.forEach(link => {
    if (link.getAttribute('href') === currentPage) {
      link.style.color = 'var(--primary)';
      link.style.fontWeight = '600';
    }
  });
  
  // Documentation sidebar active link
  if (window.location.pathname.includes('docs.html')) {
    const docLinks = document.querySelectorAll('.doc-nav a');
    docLinks.forEach(link => {
      link.addEventListener('click', function() {
        docLinks.forEach(l => l.style.fontWeight = 'normal');
        this.style.fontWeight = '600';
        this.style.color = 'var(--primary)';
      });
    });
    
    // Highlight current section based on scroll
    const sections = document.querySelectorAll('.doc-content h2[id]');
    if (sections.length > 0 && 'IntersectionObserver' in window) {
      const sectionObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            const id = entry.target.getAttribute('id');
            docLinks.forEach(link => {
              if (link.getAttribute('href') === `#${id}`) {
                docLinks.forEach(l => {
                  l.style.fontWeight = 'normal';
                  l.style.color = '';
                });
                link.style.fontWeight = '600';
                link.style.color = 'var(--primary)';
              }
            });
          }
        });
      }, {
        threshold: 0.5,
        rootMargin: '-100px 0px -50% 0px'
      });
      
      sections.forEach(section => sectionObserver.observe(section));
    }
  }
  
  // Add copy button to code blocks
  const codeBlocks = document.querySelectorAll('pre code');
  codeBlocks.forEach(block => {
    const pre = block.parentElement;
    pre.style.position = 'relative';
    
    const copyButton = document.createElement('button');
    copyButton.textContent = 'Copy';
    copyButton.type = 'button'; // Explicit button type for security
    copyButton.style.cssText = `
      position: absolute;
      top: 0.5rem;
      right: 0.5rem;
      padding: 0.25rem 0.5rem;
      background: var(--primary);
      color: white;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-size: 0.75rem;
      opacity: 0;
      transition: opacity 0.2s;
      will-change: opacity;
    `;
    
    pre.addEventListener('mouseenter', () => {
      copyButton.style.opacity = '1';
    });
    
    pre.addEventListener('mouseleave', () => {
      copyButton.style.opacity = '0';
    });
    
    copyButton.addEventListener('click', () => {
      // XSS Protection: Use textContent to get plain text only
      const text = block.textContent || '';
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(() => {
          copyButton.textContent = 'Copied!';
          setTimeout(() => {
            copyButton.textContent = 'Copy';
          }, 2000);
        }).catch(err => {
          console.error('Failed to copy:', err);
        });
      }
    });
    
    pre.appendChild(copyButton);
  });
  
  // Performance optimization: Preload critical pages
  const criticalPages = ['pricing.html', 'download.html', 'docs.html'];
  criticalPages.forEach(page => {
    const link = document.createElement('link');
    link.rel = 'prefetch';
    link.href = page;
    document.head.appendChild(link);
  });
  
  // Add loading state to external links and ensure security attributes
  const externalLinks = document.querySelectorAll('a[target="_blank"]');
  externalLinks.forEach(link => {
    // Security: Ensure all external links have rel="noopener noreferrer"
    if (!link.hasAttribute('rel') || !link.getAttribute('rel').includes('noopener')) {
      link.setAttribute('rel', 'noopener noreferrer');
    }
    
    // XSS Protection: Validate href before allowing navigation
    link.addEventListener('click', function(e) {
      const href = this.getAttribute('href');
      if (href && !isValidUrl(href)) {
        e.preventDefault();
        console.warn('Blocked potentially unsafe URL:', href);
        return false;
      }
      
      this.style.opacity = '0.6';
      setTimeout(() => {
        this.style.opacity = '1';
      }, 1000);
    });
  });
});

// Service Worker registration for offline support (optional)
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    // Uncomment to enable service worker
    // navigator.serviceWorker.register('/sw.js')
    //   .then(registration => console.log('SW registered'))
    //   .catch(err => console.log('SW registration failed'));
  });
}

// Utility function for analytics (placeholder)
function trackEvent(category, action, label) {
  // XSS Protection: Sanitize inputs before logging
  const sanitize = (str) => String(str).replace(/[<>'"]/g, '');
  console.log(`Analytics: ${sanitize(category)} - ${sanitize(action)} - ${sanitize(label)}`);
  // In production, integrate with Google Analytics, Plausible, etc.
  // if (typeof gtag !== 'undefined') {
  //   gtag('event', sanitize(action), {
  //     event_category: sanitize(category),
  //     event_label: sanitize(label)
  //   });
  // }
}

// XSS Protection: Sanitize user input
function sanitizeInput(input) {
  const div = document.createElement('div');
  div.textContent = input;
  return div.innerHTML;
}

// XSS Protection: Validate URLs to prevent javascript: protocol
function isValidUrl(url) {
  try {
    const parsed = new URL(url, window.location.origin);
    return ['http:', 'https:', 'mailto:'].includes(parsed.protocol);
  } catch {
    return false;
  }
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { trackEvent };
}
