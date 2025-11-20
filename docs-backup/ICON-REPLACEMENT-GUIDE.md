# Icon Replacement Guide

Currently, the site uses emoji icons (🔍, 🎨, ⚡, etc.). Here's how to replace them with custom images or icon fonts.

## Current Emoji Usage

Emojis are used in:
- Navbar brand: 🔍
- Feature cards: 🎨, ⚡, 🎯, 📊, 💻, 📁
- Contact page: 📧, 🐛, 💼
- Download page: 💻, 🍎
- 404 page: ⬇️, 📚, 💬

## Option 1: Replace with Image Files (Recommended)

### Step 1: Create/Download Icons

Create or download icons (PNG, SVG, or WebP):
- Size: 64x64px for card icons, 32x32px for small icons
- Format: SVG (best for scaling) or PNG with transparency
- Style: Match your brand (flat, outlined, filled, etc.)

**Free Icon Sources:**
- https://heroicons.com (SVG, MIT license)
- https://fontawesome.com (free tier)
- https://icons8.com (free with attribution)
- https://www.flaticon.com (free with attribution)
- https://iconmonstr.com (free, no attribution)

### Step 2: Add Icons to Project

Place icons in `docs/images/icons/`:
```
docs/images/icons/
├── search.svg          (navbar)
├── palette.svg         (visual similarity)
├── lightning.svg       (fast)
├── target.svg          (category filtering)
├── chart.svg           (detailed scoring)
├── desktop.svg         (desktop app)
├── folder.svg          (batch processing)
├── email.svg           (contact)
├── bug.svg             (bug reports)
├── briefcase.svg       (business)
├── windows.svg         (Windows download)
├── apple.svg           (macOS download)
├── download.svg        (download)
├── book.svg            (documentation)
└── chat.svg            (contact/support)
```

### Step 3: Update HTML

Replace emoji with image tags. Example for navbar:

**Before:**
```html
<a href="index.html" class="navbar-brand">🔍 Product Matcher</a>
```

**After:**
```html
<a href="index.html" class="navbar-brand">
  <img src="images/icons/search.svg" alt="" class="navbar-icon"> Product Matcher
</a>
```

### Step 4: Update CSS

Add icon styling:

```css
/* Navbar icon */
.navbar-icon {
  width: 24px;
  height: 24px;
  vertical-align: middle;
  margin-right: 0.5rem;
}

/* Card icons - replace emoji with image */
.card-icon img {
  width: 40px;
  height: 40px;
  filter: brightness(0) invert(1); /* Makes icon white */
}

/* Small icons */
.icon-sm {
  width: 20px;
  height: 20px;
  vertical-align: middle;
  margin-right: 0.25rem;
}
```

---

## Option 2: Use Font Awesome (Icon Font)

### Step 1: Add Font Awesome

Add to `<head>` of all HTML files:

```html
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
```

### Step 2: Replace Emojis

**Before:**
```html
<div class="card-icon">🎨</div>
```

**After:**
```html
<div class="card-icon"><i class="fas fa-palette"></i></div>
```

### Icon Mappings:

| Current Emoji | Font Awesome Class |
|---------------|-------------------|
| 🔍 | `fas fa-search` |
| 🎨 | `fas fa-palette` |
| ⚡ | `fas fa-bolt` |
| 🎯 | `fas fa-bullseye` |
| 📊 | `fas fa-chart-bar` |
| 💻 | `fas fa-desktop` |
| 📁 | `fas fa-folder` |
| 📧 | `fas fa-envelope` |
| 🐛 | `fas fa-bug` |
| 💼 | `fas fa-briefcase` |
| 🍎 | `fab fa-apple` |
| ⬇️ | `fas fa-download` |
| 📚 | `fas fa-book` |
| 💬 | `fas fa-comments` |

---

## Option 3: Use Heroicons (SVG Icons)

### Step 1: Download Icons

Go to https://heroicons.com and download the icons you need.

### Step 2: Inline SVG

Replace emoji with inline SVG:

```html
<div class="card-icon">
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
  </svg>
</div>
```

---

## Quick Replacement Script

I can create an updated version of your HTML files with any of these options. Which would you prefer?

1. **Image files** (you provide/download icons)
2. **Font Awesome** (easiest, CDN-based)
3. **Heroicons** (modern, inline SVG)
4. **Mix** (images for main icons, Font Awesome for small ones)

---

## Pros/Cons

| Method | Pros | Cons |
|--------|------|------|
| **Emojis** (current) | No files needed, universal | Inconsistent across platforms, limited styling |
| **Image Files** | Full control, custom branding | Need to create/download, more files |
| **Font Awesome** | Huge library, easy to use | External dependency, CDN required |
| **Inline SVG** | No external files, scalable | Verbose HTML, harder to maintain |

---

## My Recommendation

**For a professional site:** Use **image files** (SVG format) with your brand colors.

**For quick setup:** Use **Font Awesome** - it's the fastest and looks professional.

---

## Let Me Update the Files

Tell me which option you prefer, and I'll:
1. Update all HTML files with the new icons
2. Add necessary CSS styling
3. Create a list of icons you need to download (if using images)
4. Update the CSP if needed (for Font Awesome CDN)

Which option would you like me to implement?
