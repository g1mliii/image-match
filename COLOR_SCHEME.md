# Color Scheme Customization

## How to Change Colors (Easy!)

All colors are now controlled by **CSS variables** defined in one place.

### Quick Change Instructions

1. Open `backend/static/styles-common.css`
2. Edit the `:root` section at the top (lines 4-23)
3. Save the file
4. Refresh the app - **done!**

---

## Current Color Scheme: Warm Beige

```css
:root {
    /* Background colors */
    --bg-primary: #f5f0e6;      /* Main warm beige background */
    --bg-secondary: #ebe5d6;    /* Lighter beige for hover states */

    /* Text and border colors */
    --color-text: #000;         /* Primary text (black) */
    --color-border: #000;       /* Border color (black) */
    --color-white: #fff;        /* White (for inverted sections) */

    /* Semantic colors */
    --color-warning-bg: #f5e6cc;   /* Warning backgrounds */
    --color-warning-border: #000;  /* Warning borders */
    --color-warning-text: #856404; /* Warning text */

    /* Inverted sections */
    --bg-inverted: #000;        /* Black background */
    --color-inverted: #fff;     /* White text on black */
}
```

---

## Example Color Schemes

### 1. Original White Theme
```css
--bg-primary: #fafaf8;    /* Off-white */
--bg-secondary: #f0f0f0;  /* Light gray */
```

### 2. Dark Mode
```css
--bg-primary: #1a1a1a;       /* Dark gray */
--bg-secondary: #2a2a2a;     /* Lighter dark gray */
--color-text: #fff;          /* White text */
--color-border: #fff;        /* White borders */
--bg-inverted: #fff;         /* White background for inverted */
--color-inverted: #000;      /* Black text for inverted */
```

### 3. Cool Blue
```css
--bg-primary: #e6f2ff;    /* Light blue */
--bg-secondary: #d6e9ff;  /* Lighter blue */
```

### 4. Soft Green
```css
--bg-primary: #e8f5e9;    /* Light green */
--bg-secondary: #dcedc8;  /* Lighter green */
```

### 5. Minimal Gray
```css
--bg-primary: #f5f5f5;    /* Light gray */
--bg-secondary: #ebebeb;  /* Lighter gray */
```

---

## What Changed?

**Before:** Colors hardcoded 300+ times across 6 files
**After:** Colors defined once, used everywhere via `var(--bg-primary)` etc.

**Updated Files:**
- ✅ `backend/static/styles-common.css` - Variable definitions
- ✅ `backend/static/styles.css` - 236 references updated
- ✅ `backend/static/index.html` - Inline styles updated
- ✅ `backend/static/catalog-manager.html` - Embedded styles updated
- ✅ `backend/static/mobile-upload.html` - Embedded styles updated
- ✅ `backend/static/csv-builder.html` - Inline styles updated

---

## Technical Notes

- CSS variables work in all modern browsers (IE 11+ with fallbacks)
- Variables cascade and inherit (perfect for theming)
- No build step or preprocessor needed
- Changes take effect immediately on refresh

---

## Future Improvements

Consider adding:
- User-selectable themes (saved to localStorage)
- Automatic dark mode based on system preference
- Color contrast checker for accessibility
- Theme preview tool
