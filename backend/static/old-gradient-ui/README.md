# Old Gradient UI (Archived)

This folder contains the original gradient-based UI design that was replaced with the brutalist design.

## Files
- `index.html` - Original main app with gradient design
- `styles.css` - Original CSS with gradients, shadows, rounded corners
- `csv-builder.css` - Original CSV builder styles

## Access Old Design
To view the old gradient design, visit:
```
http://localhost:5000/gradient
```

## Why Archived?
The brutalist/minimalist design is now the default for:
- Faster loading (no web fonts, simpler CSS)
- Better accessibility (high contrast)
- Unique aesthetic
- Cleaner, more direct UX

## Restore Instructions
If you want to restore the gradient design as default:

1. Copy files from this folder back to `backend/static/`
2. Rename them to remove `-gradient` suffix
3. Update `app.py` routes accordingly

---
**Note:** The brutalist design is now the production version.
