// Initialize Lucide icons
lucide.createIcons();

// Navbar scroll effect
const navbar = document.getElementById('navbar');
window.addEventListener('scroll', () => {
  if (window.scrollY > 20) {
    navbar.classList.add('bg-white/90', 'backdrop-blur-md', 'shadow-sm', 'border-b', 'border-slate-200');
    navbar.classList.remove('bg-transparent');
  } else {
    navbar.classList.remove('bg-white/90', 'backdrop-blur-md', 'shadow-sm', 'border-b', 'border-slate-200');
    navbar.classList.add('bg-transparent');
  }
});

// Mobile menu toggle
const mobileMenuBtn = document.getElementById('mobile-menu-btn');
const mobileMenu = document.getElementById('mobile-menu');
const menuIcon = document.getElementById('menu-icon');

let isMenuOpen = false;

mobileMenuBtn.addEventListener('click', () => {
  isMenuOpen = !isMenuOpen;
  if (isMenuOpen) {
    mobileMenu.classList.remove('hidden');
    navbar.classList.add('bg-white'); // Ensure background is white when menu is open
    // Change icon to X (we'd need to re-render or swap SVG, but for simplicity we'll keep menu icon or swap class if using font icons)
    // Since we are using Lucide JS, we can't easily swap the SVG content without re-running createIcons or manual manipulation.
    // For a simple static site, toggling visibility is enough.
  } else {
    mobileMenu.classList.add('hidden');
    if (window.scrollY <= 20) {
      navbar.classList.remove('bg-white');
    }
  }
});

// Close mobile menu when clicking a link
document.querySelectorAll('#mobile-menu a').forEach(link => {
  link.addEventListener('click', () => {
    isMenuOpen = false;
    mobileMenu.classList.add('hidden');
  });
});

// Countdown Timer for Download Page
function initCountdown() {
  const windowsCountdown = document.getElementById('windows-countdown');
  const macosCountdown = document.getElementById('macos-countdown');
  
  // Only run if countdown elements exist (on download page)
  if (!windowsCountdown || !macosCountdown) return;
  
  // Set launch date to January 2, 2026 at midnight
  const launchDate = new Date('2026-01-02T00:00:00').getTime();
  
  function updateCountdown() {
    const now = new Date().getTime();
    const distance = launchDate - now;
    
    // Calculate time units
    const days = Math.floor(distance / (1000 * 60 * 60 * 24));
    const hours = Math.floor((distance % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
    const minutes = Math.floor((distance % (1000 * 60 * 60)) / (1000 * 60));
    const seconds = Math.floor((distance % (1000 * 60)) / 1000);
    
    // Display countdown
    const countdownText = `Available in: ${days}d ${hours}h ${minutes}m ${seconds}s`;
    
    if (windowsCountdown) windowsCountdown.textContent = countdownText;
    if (macosCountdown) macosCountdown.textContent = countdownText;
    
    // If countdown is finished
    if (distance < 0) {
      if (windowsCountdown) windowsCountdown.textContent = 'Available Now!';
      if (macosCountdown) macosCountdown.textContent = 'Available Now!';
      clearInterval(countdownInterval);
    }
  }
  
  // Update immediately and then every second
  updateCountdown();
  const countdownInterval = setInterval(updateCountdown, 1000);
}

// Initialize countdown when DOM is loaded
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initCountdown);
} else {
  initCountdown();
}
