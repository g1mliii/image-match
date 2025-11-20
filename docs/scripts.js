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
